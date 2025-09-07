import os
import re
from time import sleep
from typing import Tuple

from .llm import LLM


class DeepSeekModelAPI(LLM):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name

        # Lazily create client to allow env var setup before first call
        self._client = None

    def _get_client(self):
        if self._client is None:
            # Prefer dedicated DeepSeek env vars; fallback to OpenAI ones for convenience
            api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = (
                os.getenv("DEEPSEEK_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.deepseek.com"
            )
            # OpenAI 1.x style client
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def _build_messages(self, system_prompt, dialog):
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        messages = []
        input_text = ""
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            input_text += system_prompt
        for d in dialog:
            role = d["role"]
            # 将 assistant_start 作为用户指令注入，保持最后一条消息为 user，利于模型对齐
            if role == "assistant_start":
                role = "user"
            input_text += "\n" + role + ": "
            input_text += d["content"]
            messages.append({"role": role, "content": d["content"]})
        return input_text, messages

    def _create_chat_completion(self, messages, max_tokens: int, temperature: float = 0.1, top_p: float = 0.9):
        client = self._get_client()
        fail_count = 0
        while True:
            try_again = False
            try:
                resp = client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=30,  # 添加30秒超时
                )
            except Exception as e:
                print(f"⚠️ [DeepSeek] API调用失败 (尝试 {fail_count + 1}): {e}")
                sleep(10)
                try_again = True
                fail_count += 1
                if fail_count >= 3:  # 最多重试3次
                    print("❌ [DeepSeek] 达到最大重试次数，退出")
                    raise e
            if not try_again:
                return resp

    def request_rating_0_9(self, system_prompt, dialog) -> Tuple[str, str]:
        input_text, messages = self._build_messages(system_prompt, dialog)
        # 只需要输出一个数字，设置较小的max_tokens
        out = self._create_chat_completion(messages, max_tokens=5, temperature=0.2, top_p=0.9)
        raw_response = out.choices[0].message.content.strip()
        
        # 改进解析逻辑：只匹配0-9的单个数字
        # 如果响应中包含"10"等超出范围的数字，应该被拒绝
        rating_match = re.search(r'\b([0-9])\b', raw_response)
        if rating_match:
            extracted_rating = rating_match.group(1)
            return input_text, extracted_rating
        else:
            # 检查是否包含超出范围的数字
            return input_text, "5"  # 默认中等评分

    def request_rating_1_10(self, system_prompt, dialog) -> Tuple[str, str]:
        input_text, messages = self._build_messages(system_prompt, dialog)
        # 只需要输出一个数字，设置较小的max_tokens
        out = self._create_chat_completion(messages, max_tokens=5, temperature=0.2, top_p=0.9)
        raw_response = out.choices[0].message.content.strip()
        
        # 改进解析逻辑：提取第一个数字1-10
        rating_match = re.search(r'\b(10|[1-9])\b', raw_response)
        if rating_match:
            extracted_rating = rating_match.group(1)
            return input_text, extracted_rating
        else:
            return input_text, "5"  # 默认中等评分

    def request_rating_1_5(self, system_prompt, dialog) -> Tuple[str, str]:
        input_text, messages = self._build_messages(system_prompt, dialog)
        
        # 显示发送给API的完整请求信息
        print(f"\n🚀 [DeepSeek API] 发送评分请求 (1-5):")
        print(f"📊 模型: {self.name}")
        print(f"📝 输入文本长度: {len(input_text)} 字符")
        print(f"💬 消息数量: {len(messages)}")
        
        # 显示完整的请求内容
        print(f"\n📋 发送给API的完整内容:")
        print("="*80)
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            print(f"[{i+1}] {role.upper()}:")
            print(content)
            print("-" * 40)
        print("="*80)
        
        # 只需要输出一个数字，设置较小的max_tokens
        out = self._create_chat_completion(messages, max_tokens=5, temperature=0.2, top_p=0.9)
        raw_response = out.choices[0].message.content.strip()
        
        # 改进解析逻辑：提取第一个数字1-5
        rating_match = re.search(r'\b([1-5])\b', raw_response)
        if rating_match:
            extracted_rating = rating_match.group(1)
            return input_text, extracted_rating
        else:
            return input_text, "3"  # 默认中等评分

    def request_rating_text(self, system_prompt, dialog) -> Tuple[str, str]:
        input_text, messages = self._build_messages(system_prompt, dialog)
        out = self._create_chat_completion(messages, max_tokens=8, temperature=0.2, top_p=0.9)
        res = out.choices[0].message.content
        return input_text, res

    def query(self, prompt: str) -> str:
        """通用查询方法，用于测试API连接"""
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = self._create_chat_completion(messages, max_tokens=50, temperature=0.1, top_p=0.9)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ [DeepSeek] 查询失败: {e}")
            return f"错误: {e}"

    def request_explanation(self, system_prompt, dialog) -> Tuple[str, str]:
        input_text, messages = self._build_messages(system_prompt, dialog)
        # 调整参数，确保生成简洁但完整的解释
        out = self._create_chat_completion(messages, max_tokens=50, temperature=0.5, top_p=0.9)
        res = out.choices[0].message.content
        return input_text, res


