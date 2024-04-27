from fastapi import FastAPI
from modal import asgi_app, method
from apicontainer import stub
from tools import generator
import torch

web_app = FastAPI()
TIMEOUT = 2
GPU = "A10G"

# Ressourcenparameter, es sind auch mehrere GPU möglich
@stub.cls(gpu=GPU, container_idle_timeout=TIMEOUT)
class Model:
    def __init__(self, sched, LoRA):
        import torch
        from diffusers import DiffusionPipeline
        from compel import Compel, ReturnedEmbeddingsType

        # Muss noch vor dem Modell geladen werden, weil es davon abhängig ist
        self.scheduler = self.get_right_sched(sched)

        # Einstellungen die Modell und Refiner betreffen
        # 16 bit VAE müsste dann auch hier rein. Müsste mal getestet werden
        # 16 bit VAE evtl. zu einer nutzerentscheidung machen? Custom VAE support?
        load_options = dict(
            scheduler=self.scheduler,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Lädt das Basismodell
        # Die Pipeline sollte evtl. auseinandergenommen werden für mehr Konfigurationsmöglichkeiten
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

        # Lädt den Refiner
        # TODO: Option zum überspringen ist für einige LoRA teilweise nötig (Pixel)
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

        # LoRA werden zwischen die Schichten geschoben, daher erst jetzt möglich
        self.apply_lora(LoRA)

        # Compbel übernimmt das Umwandeln von Token/Worte in Embeddings um Gewichte setzen zu könen
        # Dude with a large hat -> Dude with a large++ hat -> Dude with a large(1.5) hat
        # 2.0 bedeutet aber nicht doppelt so wichtig. Das ist fast schon ein subjektiver Parameter
        self.compel = Compel(tokenizer=[self.base.tokenizer, self.base.tokenizer_2],
                        text_encoder=[self.base.text_encoder, self.base.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=[False, True])


    def apply_lora(self, name):
        """
        Schiebt das LoRA ins Modell
        """
        from safetensors.torch import load_file
        cache_path = "/vol/cache"

        if name == "Nichts":
            return
        else:
            lora_path = f"{cache_path}/LoRA/{name}.safetensors"
        lora_state_dict = load_file(lora_path)
        self.base.load_lora_weights(lora_state_dict)


    def get_right_sched(self, name):
        """
        Abhängig vom Parameter wird der richtige Scheduler gewählt, geladen und zurückgegeben
        """
        from diffusers import DPMSolverMultistepScheduler, DEISMultistepScheduler, CMStochasticIterativeScheduler, EulerAncestralDiscreteScheduler
        cache_path = "/vol/cache"

        # Um Code zu reduzieren ein Key Value storage
        # Key: Namen der Methoden
        # Value: Tuple aus dem Objekt und einem weiteren dictionary mit den dazugehörend benötigten Parametern
        schedulers = {
            "DPM++2M": (DPMSolverMultistepScheduler, {"solver_type": "midpoint", "algorithm_type": "dpmsolver++"}),
            "DPM++2M-Karras": (DPMSolverMultistepScheduler, {"solver_type": "midpoint", "algorithm_type": "dpmsolver++", "use_karras_sigmas": True}),
            "DEIS": (DEISMultistepScheduler, {"algorithm_type": "deis"}),
            "CMS_TEST": (CMStochasticIterativeScheduler, {}),
            "Euler-a": (EulerAncestralDiscreteScheduler, {}),
        }

        scheduler_class, params = schedulers[name]
        # Hier werden dann auch alle anderen Parameter eingefügt die immer nötig sind, oder zumindest ignoriert werden
        return scheduler_class.from_pretrained(cache_path, subfolder="scheduler", solver_order=2, prediction_type="epsilon", device_map="auto", **params)


    @method()
    def inference(self, prompt, negative_prompt, batch_size, steps, fraq, guidance, rand_val, height=1024, width=1024):
        conditioning, pooled = self.compel(prompt)
        conditioning1, pooled1 = self.compel(negative_prompt)
        gen_list = generator(batch_size, rand_val)

        torch.cuda.empty_cache()
        with torch.no_grad():
            image = self.base(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=conditioning1,
                negative_pooled_prompt_embeds=pooled1,
                generator=gen_list,
                num_inference_steps=steps,
                denoising_end=fraq,
                num_images_per_prompt=batch_size,
                guidance_scale=guidance,
                height=height,
                width=width,
                output_type="latent",
            ).images
            torch.cuda.empty_cache()
            image = self.refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=gen_list,
                num_inference_steps=steps,
                denoising_start=fraq,
                num_images_per_prompt=batch_size,
                guidance_scale=guidance,
                image=image,
            ).images
        return image


@stub.function()
@asgi_app(label="api-sdxl-gundo")
def app():

    @web_app.get("/infer/{prompt}")
    async def infer(prompt: str):
        from fastapi.responses import Response
        import io

        t_prompt, t_negative_prompt = prompt.split("|")

        dpm, lora = "DPM++2M-Karras", "Nichts"
        t_negative_prompt = t_negative_prompt + ", broken, error, text, letters, numbers, nsfw, nude, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck"
        batch_size, steps, fraq, guidance, rand_val = 1, 48, 0.88, 9, ""
        model = Model(dpm, lora)
        image = model.inference.remote(t_prompt, t_negative_prompt, batch_size, steps, fraq, guidance, rand_val, 832, 1248)
        image = image[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return Response(image_bytes, media_type="image/png")

    return web_app