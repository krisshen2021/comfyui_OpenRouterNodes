import base64
import requests
from PIL import Image
import io
import numpy as np
import yaml
import torch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
default_base_url = "https://openrouter.ai/api/v1"


# 自定义节点类
class OpenRouterOAINode_Models:
    _cached_model_list = None

    @classmethod
    def INPUT_TYPES(cls):
        if cls._cached_model_list is None:
            cls.load_model_list()
        return {
            "required": {
                "model": (
                    cls._cached_model_list,
                    {"default": cls._cached_model_list[0]},
                )
            }
        }

    RETURN_TYPES = ("STRING",)  # 输出类型
    RETURN_NAMES = ("model",)  # 输出名称
    FUNCTION = "process"  # 指定处理函数
    CATEGORY = "OpenRouter"  # 节点分类

    @classmethod
    def load_model_list(cls):
        """
        加载模型列表并缓存
        """
        try:
            # 请求模型列表
            models_res = requests.get(f"{default_base_url}/models")
            if models_res.status_code != 200:
                raise Exception(
                    f"API Error: {models_res.status_code}, {models_res.text}"
                )
            model_data = models_res.json()
            cls._cached_model_list = [model["id"] for model in model_data["data"]]
            # print(f"Loaded models: {cls._cached_model_list}")
        except Exception as e:
            # 如果加载失败,使用默认模型列表
            cls._cached_model_list = ["default_model"]
            print(
                f"Failed to load models, using default list: {cls._cached_model_list}. Error: {e}"
            )

    def process(self, model):
        return (model,)


class OpenRouterOAINode_hunyuanPrompt:
    """
    OpenRouter hunyuan video prompt generator
    """

    _yaml_path = os.path.join(script_dir, "prompt", "hunyuan.yaml")
    _yaml_data = None
    _system_prompt = None

    @classmethod
    def read_yaml_prompts(cls):
        try:
            with open(cls._yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data
        except FileNotFoundError:
            print(f"Error:File not found at{cls._yaml_path}")
            return None

    @classmethod
    def INPUT_TYPES(cls):
        if cls._yaml_data is None:
            cls._yaml_data = cls.read_yaml_prompts()
            # print(f"Loaded prompts: {cls._prompt_list}")
            cls._system_prompt = cls._yaml_data["Prompt"]
        return {
            "optional": {
                "video_description": ("STRING", {"default": None, "multiline": True,}),
            }
        }

    RETURN_TYPES = ("STRING",)  # 输出类型
    RETURN_NAMES = ("Hunyuan_video_prmopt",)  # 输出名称
    FUNCTION = "process"  # 指定处理函数
    CATEGORY = "OpenRouter"  # 节点分类

    def process(self, video_description = None):  # 处理函数
        if video_description:
            print("video description", video_description)
            Hunyuan_video_prompt = self.__class__._system_prompt + f"\ngiven input:\ninput: '{video_description}'"
        else:
            Hunyuan_video_prompt = self.__class__._system_prompt
        return (Hunyuan_video_prompt,)


class OpenRouterOAINode_txt2imgPrompt:
    """
    OpenRouter txt2img prompt node
    """

    _yaml_path = os.path.join(script_dir, "prompt", "text2image.yaml")
    _prompt_list = None
    _prompt_keys = []

    @classmethod
    def INPUT_TYPES(cls):
        if cls._prompt_list is None:
            cls._prompt_list = cls.read_yaml_prompts()
            # print(f"Loaded prompts: {cls._prompt_list}")
            cls._prompt_keys = list(cls._prompt_list.keys())
        return {
            "required": {
                "Diffusion_Model": (cls._prompt_keys, {"default": cls._prompt_keys[0]}),
            }
        }

    @classmethod
    def read_yaml_prompts(cls):
        try:
            with open(cls._yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data
        except FileNotFoundError:
            print(f"Error:File not found at{cls._yaml_path}")
            return None

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )  # 输出类型
    RETURN_NAMES = (
        "Positive_prompt_suffix",
        "Negative_prompt_suffix",
    )  # 输出名称
    FUNCTION = "process"  # 指定处理函数
    CATEGORY = "OpenRouter"  # 节点分类

    def process(self, Diffusion_Model):  # 处理函数
        Positive_prompt_suffix = f"\nONLY create positive prompt of the {Diffusion_Model} diffusion model base on giving context, below is the Guide of how to create for {Diffusion_Model} diffusion models, only output the final result:\nPositive prompt template:\n{self.__class__._prompt_list[Diffusion_Model]['Positive_Template']}\nPositive prompt example:\n{self.__class__._prompt_list[Diffusion_Model]['Positive_Prompt']}\n"
        Negative_prompt_suffix = f"\nONLY create negative prompt of {Diffusion_Model} diffusion model according to giving context, do not add too much restrict prompt, keep essential, below is the Guide of how to create for {Diffusion_Model} diffusion models, only output the final result:\nNegative prompt template:\n{self.__class__._prompt_list[Diffusion_Model]['Negative_Template']}\nNegative prompt example:\n{self.__class__._prompt_list[Diffusion_Model]['Negative_Prompt']}\n"
        return (
            Positive_prompt_suffix,
            Negative_prompt_suffix,
        )


class OpenRouterOAINode_Infer:

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入参数类型。
        """

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "system_prompt": ("STRING", {"multiline": True}),
                "api_base_url": (
                    "STRING",
                    {"default": default_base_url},
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "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"
                    },
                ),  # OpenRouter API Key
                "model": (
                    "STRING",
                    {"default": None, "multiline": False, "defaultInput": True},
                ),
                # 模型名称
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.5, "max": 2.0, "step": 0.05},
                ),
            },
            "optional": {
                "prompt_input": (
                    "STRING",
                    {"default": None, "multiline": True, "defaultInput": True},
                ),
                "system_prompt_input": (
                    "STRING",
                    {"default": None, "multiline": True, "defaultInput": True},
                ),
                "max_token": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 512,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "image": ("IMAGE",),
                "attach_image": ("BOOLEAN", {"default": False}),
                "image_max_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 512,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)  # 输出类型
    RETURN_NAMES = ("Response",)  # 输出名称
    FUNCTION = "process"  # 指定处理函数
    CATEGORY = "OpenRouter"  # 节点分类

    def process(
        self,
        prompt,
        system_prompt,
        api_base_url,
        api_key,
        model,
        temperature,
        prompt_input=None,
        system_prompt_input=None,
        max_token=512,
        image=None,
        attach_image=False,
        image_max_size=1024,
    ):
        """
        处理逻辑: 将文本和图像组合发送到 OpenRouter API。
        """
        prompt = prompt_input or prompt
        system_prompt = system_prompt_input or system_prompt

        if image is not None and attach_image:
            print("Image tensor shape:", image.shape)
            print("Image tensor dtype:", image.dtype)
            if image.shape[0] != 1:
                raise ValueError("Input image must have batch size of 1.")
            image_tensor = image.squeeze(0)  # 移除批量维度
            print("image tensor after squeeze", image_tensor.shape)
            # 获取图像的高度和宽度
            height, width = image_tensor.shape[0], image_tensor.shape[1]
            print(f"Original image size: height={height}, width={width}")
            # 判断是否需要缩放
            max_size = image_max_size
            if height > max_size or width > max_size:
                print(
                    f"Image too large, resizing to fit within {max_size} while maintaining aspect ratio."
                )
                # 计算缩放比例
                scale = min(max_size / height, max_size / width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                if image_tensor.shape[-1] == 3:  # 如果通道维度在最后
                    image_tensor = image_tensor.permute(
                        2, 0, 1
                    )  # 转换为 [Channels, Height, Width]
                    print("Image tensor shape after permute:", image.shape)
                # 使用双线性插值进行缩放
                image_tensor = (
                    torch.nn.functional.interpolate(
                        image_tensor.unsqueeze(0),  # 增加批量维度 [1, C, H, W]
                        size=(new_height, new_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )  # 移除批量维度 [H, W, C]
                print(f"Resized image size: height={new_height}, width={new_width}")

            # 转换数据类型为 uint8, 并缩放到 [0, 255] 范围
            image_tensor = (image_tensor * 255).cpu().numpy().astype("uint8")
            print("Image tensor shape after scale", image_tensor.shape)
            print("Image tensor dtype after scale", image_tensor.dtype)
            image_pil = Image.fromarray(image_tensor)
            buffered = io.BytesIO()
            image_pil.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 构造 OpenAI 消息格式
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                },
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        # API 请求头和数据
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_token,
        }

        # 发送请求到 OpenRouter API
        response = requests.post(
            f"{api_base_url}/chat/completions", headers=headers, json=data
        )

        # 检查响应状态
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code}, {response.text}")

        # 解析返回的消息
        response_data = response.json()
        print("response is", response_data)
        response_text = response_data["choices"][0]["message"]["content"]

        return (response_text,)
