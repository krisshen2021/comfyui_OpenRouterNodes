from .openrouter_oai_node import OpenRouterOAINode_Infer
from .openrouter_oai_node import OpenRouterOAINode_Models
from .openrouter_oai_node import OpenRouterOAINode_txt2imgPrompt
from .openrouter_oai_node import OpenRouterOAINode_hunyuanPrompt

NODE_CLASS_MAPPINGS = {
    "OpenRouterOAINode_Infer": OpenRouterOAINode_Infer,
    "OpenRouterOAINode_Models": OpenRouterOAINode_Models,
    "OpenRouterOAINode_txt2imgPrompt": OpenRouterOAINode_txt2imgPrompt,
    "OpenRouterOAINode_hunyuanPrompt": OpenRouterOAINode_hunyuanPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterOAINode_Infer": "OpenRouter OAI Node Infer",
    "OpenRouterOAINode_Models": "OpenRouter OAI Node Models",
    "OpenRouterOAINode_txt2imgPrompt": "OpenRouter OAI Node txtimgPrompt",
    "OpenRouterOAINode_hunyuanPrompt": "OpenRouter OAI Node hunyuanPrompt"
}