import warnings
warnings.filterwarnings("ignore")

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
#os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
import numpy as np
import random
from PIL import Image

from modeling_dit_wikiart import DiTWikiartModel
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler

# --- IMPORTAÇÃO DO MAPPING ---
try:
    from mapping import reverse_genre_mapping, reverse_style_mapping
except ImportError:
    print("ERRO: O arquivo 'mapping.py' não foi encontrado. Certifique-se de que ele está na mesma pasta.")
    exit()

# --- CONFIGURAÇÃO DE DIRETÓRIO ---
BASE_DIR = "outputs"
VAE_SCALING_FACTOR = 0.18215

def get_unique_filename(base_path, filename):
    """Verifica se o arquivo já existe e adiciona incrementador (_1, _2...)"""
    filepath = os.path.join(base_path, f"{filename}.png")
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(base_path, f"{filename}_{counter}.png")
        counter += 1
    return filepath

def configurar_parametros():
    """Faz as perguntas iniciais e retorna as configurações escolhidas."""
    print("\n=== CONFIGURAÇÕES GERAIS ===")
    
    # 0. Escolha do Modelo
    model_str = input("Escolha o Modelo (1 - Small, 2 - Base, 3 - Large) [Enter para Small]: ").strip()
    if model_str == "2":
        model_name = "kaupane/DiT-Wikiart-Base"
        model_size = "B"
    elif model_str == "3":
        model_name = "kaupane/DiT-Wikiart-Large"
        model_size = "L"
    else:
        model_name = "kaupane/DiT-Wikiart-Small"
        model_size = "S"

    # 1. Escolha do Scheduler
    sched_str = input("Escolha o Scheduler (1 - DDPM, 2 - DDIM) [Enter para DDPM]: ").strip()
    if sched_str == "2":
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        sched_name = "DDIM"
    else:
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        sched_name = "DDPM"
        
    # 2. Seed
    seed_str = input("Digite a Seed (ou Enter para aleatória): ").strip()
    global_seed = int(seed_str) if seed_str else None
    
    # 3. Steps
    steps_str = input("Digite os Steps (ou Enter para o padrão 50): ").strip()
    global_steps = int(steps_str) if steps_str else 50
    
    # 4. CFG (Classifier-Free Guidance)
    print("\n--- CFG (Melhora a aderência ao estilo, mas usa mais VRAM) ---")
    cfg_str = input("Usar CFG? (1 - Sim, 2 - Não) [Enter para Não]: ").strip()
    use_cfg = True if cfg_str == "1" else False
    
    cfg_scale = 2.5
    if use_cfg:
        cfg_val_str = input("Qual valor do CFG? [Enter para 2.5]: ").strip()
        cfg_scale = float(cfg_val_str) if cfg_val_str else 2.5
    
    print(f"\n-> Modelo: {model_size}")
    print(f"-> Scheduler: {sched_name}")
    print(f"-> Seed: {'Aleatória' if global_seed is None else global_seed}")
    print(f"-> Steps: {global_steps}")
    print(f"-> CFG: {'Ativado (Escala: ' + str(cfg_scale) + ')' if use_cfg else 'Desativado'}")
    
    return model_name, model_size, scheduler, sched_name, global_seed, global_steps, use_cfg, cfg_scale

def gerar_imagem(model, vae, scheduler, device, g_idx, s_idx, seed, num_steps, use_cfg, cfg_scale):
    """Executa o loop de difusão usando o Scheduler escolhido."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        
    generator = torch.Generator(device=device).manual_seed(seed)
    
    g = torch.tensor([g_idx], device=device, dtype=torch.long)
    s = torch.tensor([s_idx], device=device, dtype=torch.long)
    
    # As condições nulas usam os índices máximos do modelo (42 e 137)
    g_null = torch.tensor([model.num_genres], device=device, dtype=torch.long)
    s_null = torch.tensor([model.num_styles], device=device, dtype=torch.long)
    
    scheduler.set_timesteps(num_steps)
    shape = (1, 4, 32, 32)
    latents = torch.randn(shape, device=device, generator=generator)
    
    with torch.no_grad():
        for t in scheduler.timesteps:
            t_device = t.to(device)
            
            if use_cfg:
                # Previsão condicionada e não condicionada
                eps_cond = model(latents, t_device.unsqueeze(0), g, s)
                eps_uncond = model(latents, t_device.unsqueeze(0), g_null, s_null)
                # Mistura com CFG
                predicted_noise = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                # Previsão padrão sem CFG
                predicted_noise = model(latents, t_device.unsqueeze(0), g, s)
                
            latents = scheduler.step(predicted_noise, t, latents, generator=generator).prev_sample

        latents = latents / VAE_SCALING_FACTOR
        image = vae.decode(latents).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    
    if device == "cuda":
        torch.cuda.empty_cache()
        
    return Image.fromarray(image[0]), seed

def obter_nomes(g_idx, s_idx):
    """Traduz os índices para nomes limpos seguindo as regras definidas."""
    # Gênero: Tudo mapeado do 0 ao 42 (42 = yakusha-e)
    g_name = reverse_genre_mapping.get(g_idx, f"desconhecido-{g_idx}")
    g_name = g_name.lower().replace(" ", "-").replace("(", "").replace(")", "").replace("\xa0", "-")
    
    # Estilo: 0 a 136 no mapping. 137 é a Condição Nula.
    if s_idx == 137:
        s_name = "condicao-nula"
    else:
        s_name = reverse_style_mapping.get(s_idx, f"desconhecido-{s_idx}")
        s_name = s_name.lower().replace(" ", "-").replace("(", "").replace(")", "").replace("\xa0", "-")
        
    return g_name, s_name

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
    # Carrega as configurações iniciais
    model_name, model_size, scheduler, sched_name, global_seed, global_steps, use_cfg, cfg_scale = configurar_parametros()

    print(f"\nCarregando Modelo ({model_size}) e VAE...")
    # O modelo é baixado automaticamente aqui caso não exista no cache
    model = DiTWikiartModel.from_pretrained(model_name).to(device)
    model.eval()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    print("Tudo carregado!")

    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("1 - Modo Manual")
        print("2 - Todas as combinações possíveis")
        print("3 - Modo Aleatório")
        print("4 - Alterar Configurações (Modelo/Seed/Steps/Scheduler/CFG)")
        print("0 - Sair")
        modo = input("Escolha o modo: ")

        if modo == "0":
            print("Saindo...")
            break

        elif modo == "4":
            old_model_name = model_name
            model_name, model_size, scheduler, sched_name, global_seed, global_steps, use_cfg, cfg_scale = configurar_parametros()
            
            # Se o usuário trocou o modelo, recarrega
            if model_name != old_model_name:
                print(f"\nTrocando modelo para {model_size}...")
                del model # Remove o modelo antigo da memória
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                model = DiTWikiartModel.from_pretrained(model_name).to(device)
                model.eval()
                print("Novo modelo carregado!")
            continue

        elif modo == "1":
            print("\n--- MODO MANUAL ---")
            print("AVISO SOBRE OS ÍNDICES:")
            print("-> Gêneros: 0 (None), 1 a 42 (Reais).")
            g_input = int(input("Digite o Gênero (0 a 42): "))
            
            # Confirmação visual do Gênero
            g_display = reverse_genre_mapping.get(g_input, "Desconhecido")
            print(f"-> Genero: {g_display.title()}")
            
            print("\n-> Estilos: 0 a 136 (Reais), 137 (Condição Nula).")
            s_input = int(input("Digite o Estilo (0 a 137): "))
            
            # Confirmação visual do Estilo
            if s_input == 137:
                print("-> Estilo: Condição Nula")
            else:
                s_display = reverse_style_mapping.get(s_input, "Desconhecido")
                print(f"-> Estilo: {s_display.title()}")
            
            qtd = int(input("\nQuantas imagens gerar para esta combinação? "))
            
            g_name, s_name = obter_nomes(g_input, s_input)
            pasta_saida = os.path.join(BASE_DIR, "modo_manual", f"{g_name}-{g_input}")
            os.makedirs(pasta_saida, exist_ok=True)
            
            for _ in range(qtd):
                img, actual_seed = gerar_imagem(model, vae, scheduler, device, g_input, s_input, global_seed, global_steps, use_cfg, cfg_scale)
                
                # Nome inclui a letra do modelo (S, B ou L)
                nome_base = f"{g_name}-{g_input}_{s_name}-{s_input}_{model_size}_{sched_name.lower()}_s-{global_steps}_seed-{actual_seed}"
                caminho_final = get_unique_filename(pasta_saida, nome_base)
                img.save(caminho_final)
                print(f"Salvo: {caminho_final}")

        elif modo == "2":
            print("\n--- MODO TODAS AS COMBINAÇÕES ---")
            print("São 43 Gêneros (0 a 42) x 138 Estilos (0 a 137).")
            print("Total: 5.934 imagens. Isso vai demorar um bom tempo.")
            
            confirm = input("Tem certeza que deseja continuar? (s/n): ")
            if confirm.lower() == 's':
                total = 43 * 138
                count = 0
                
                for g in range(43):
                    for s in range(138):
                        count += 1
                        g_name, s_name = obter_nomes(g, s)
                        pasta_saida = os.path.join(BASE_DIR, "modo_combinacoes", f"{g_name}-{g}")
                        os.makedirs(pasta_saida, exist_ok=True)
                        
                        img, actual_seed = gerar_imagem(model, vae, scheduler, device, g, s, global_seed, global_steps, use_cfg, cfg_scale)
                        
                        nome_base = f"{g_name}-{g}_{s_name}-{s}_{model_size}_{sched_name.lower()}_s-{global_steps}_seed-{actual_seed}"
                        caminho_final = get_unique_filename(pasta_saida, nome_base)
                        img.save(caminho_final)
                        
                        if count % 50 == 0 or count == total:
                            print(f"Progresso: [{count}/{total}] imagens geradas...")

        elif modo == "3":
            print("\n--- MODO ALEATÓRIO ---")
            qtd = int(input("Quantas imagens aleatórias deseja gerar? "))
            
            for _ in range(qtd):
                g = random.randint(0, 42)
                s = random.randint(0, 137)
                
                g_name, s_name = obter_nomes(g, s)
                pasta_saida = os.path.join(BASE_DIR, "modo_aleatorio", f"{g_name}-{g}")
                os.makedirs(pasta_saida, exist_ok=True)
                
                img, actual_seed = gerar_imagem(model, vae, scheduler, device, g, s, global_seed, global_steps, use_cfg, cfg_scale)
                
                nome_base = f"{g_name}-{g}_{s_name}-{s}_{model_size}_{sched_name.lower()}_s-{global_steps}_seed-{actual_seed}"
                caminho_final = get_unique_filename(pasta_saida, nome_base)
                img.save(caminho_final)
                print(f"Salvo: {caminho_final}")

if __name__ == "__main__":
    main()