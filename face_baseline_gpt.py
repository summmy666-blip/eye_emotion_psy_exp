import os
import base64
import requests
import json
import pandas as pd
import re
import time
import random
from tqdm import tqdm
from typing import List, Dict


class EmotionAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def encode_image(self, image_path: str) -> str:
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"图像编码错误: {e}")
            return None

    def process_single_image(self, image_path: str, emotion_label: str) -> Dict:
        encoded_image = self.encode_image(image_path)
        if encoded_image is None:
            return None

        # 获取中文形容词描述 - 修改后的提示
        adj_prompt = "这张图片包含两个面部表情。左侧面部展示的是基线表情（中性表情），右侧面部展示的是情绪表情。请参考左侧的基线表情，分析右侧面部表达了什么神情？明确用两个字的形容词表述，不要有多余的话，另外不要回答无法判断，实在判断不了你就随便说"

        # 获取情绪判断 - 修改后的提示
        emotion_prompt = """
        你是一名情绪识别专家。你将看到一张包含两个面部表情的图像：
        左侧面部展示的是基线表情（中性表情），右侧面部展示的是情绪表情。
        你的任务是参考左侧的基线表情，识别右侧面部表达的情绪。
        **重要：** 你必须从以下词语中选择一个且仅一个作为你的答案：
        angry, disgusted, fearful, happy, sad, surprised
        绝对不要回答其他任何词语！不要说"unknown"。不要提供任何解释。
        """

        # 获取情绪强度 - 修改后的提示
        intensity_prompt = """
        你是一位情绪识别专家。你将看到一个包含两张面孔的图像：
        上方图片展示的是基线表情（中性表情），下方图片展示的是同一个人表达的情绪表情。
        你的任务是参考上方的基线表情，识别下方的情绪表情传达的情绪有多强。
        **重要提示：**您的答案必须是一个介于1.00和5.00之间的数字，精确到小数点后2位。
        例如：2.75、3.90、4.25等。除非你认为必要，否则不要使用.00作为小数点。
        这些数字代表了从非常弱（1）到非常强（5）的等级。
        只提供数字，没有其他文字或解释。
        示例:2.75
        """

        # 获取情绪效价 - 修改后的提示
        valence_prompt = """
        你是一位情绪识别专家。你将看到一个包含两张面孔的图像：
        上方图片展示的是基线表情（中性表情），下方图片展示的是同一个人表达的情绪表情。
        你的任务是参考上方的基线表情，分析下方的情绪表情传达的情绪属于什么效价。
        **重要提示：**您的答案必须是一个介于1.00和5.00之间的数字，精确到小数点后2位。
        例如：2.75、3.90、4.25等。除非你认为必要，否则不要使用.00作为小数点。
        这些数字表示从非常积极（1）到较为积极（2）到中性（3）到较为消极（4）到非常消极（5）的范围。 
        只提供数字，没有其他文字或解释。
        示例:2.75
        """

        results = {}
        valid_emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]

        # 依次处理四种不同的分析
        for prompt_type, prompt in [
            ("adj", adj_prompt),
            ("emotion", emotion_prompt),
            ("intensity", intensity_prompt),
            ("valence", valence_prompt)
        ]:
            api_success = False
            max_retries_prompt_type = 5
            if prompt_type == "emotion":
                max_retries_prompt_type = 1000

            for attempt in range(max_retries_prompt_type):
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded_image}"
                                    }
                                }
                            ]
                        }
                    ]
                    response = self.session.post(
                        "https://api.openai.com/v1/chat/completions",
                        json={
                            "model": "gpt-4o",
                            "messages": messages,
                            "max_tokens": 150,
                        },
                        timeout=60
                    )

                    if response.status_code == 200:
                        response_json = response.json()
                        result = response_json['choices'][0]['message']['content'].strip()

                        if prompt_type == "emotion":
                            if result in valid_emotions:
                                results["trial_selection"] = result
                                results["result"] = 1 if result == emotion_label else 0
                                api_success = True
                                break
                            else:
                                print(f"Warning: Model returned unexpected emotion: {result}, retrying...")
                        elif prompt_type in ["intensity", "valence"]:
                            def is_valid_decimal(s):
                                try:
                                    num = float(s)
                                    if 1.0 <= num <= 5.0:
                                        decimal_part = str(s).split('.')[1] if '.' in str(s) else ''
                                        return len(decimal_part) == 2
                                    return True
                                except ValueError:
                                    return False

                            if is_valid_decimal(result):
                                if prompt_type == "intensity":
                                    results["strength"] = result
                                else:
                                    results["valence"] = result
                                api_success = True
                                break
                            else:
                                print(f"Warning: Model returned unexpected output: {result}, retrying...")
                        elif prompt_type == "adj":
                            results["textbox"] = result
                            api_success = True
                            break
                    else:
                        print(f"API请求失败，状态码: {response.status_code}，尝试重试...")

                except requests.exceptions.RequestException as e:
                    print(f"请求异常 (第 {attempt + 1} 次尝试): {e}")

                if not api_success:
                    time.sleep(1)

            if not api_success and prompt_type == "emotion":
                print(f"多次重试情绪识别失败，使用随机情绪作为结果。图像: {image_path}")
                results["trial_selection"] = random.choice(valid_emotions)
                results["result"] = 0

        # 设置情绪类型映射
        emotion_type_map = {
            "angry": "1",
            "disgusted": "2",
            "fearful": "3",
            "happy": "4",
            "sad": "5",
            "surprised": "6"
        }

        # 确保 trial_selection 存在后再进行映射，如果极其异常情况下 trial_selection 仍然不存在，则 type 默认为 "unknown"，但这在正常逻辑下不应发生
        trial_selection_result = results.get("trial_selection")
        emotion_type = "unknown"
        if trial_selection_result: # 只有当 trial_selection 存在时才进行映射
            emotion_type = emotion_type_map.get(trial_selection_result, "unknown")

        results.update({
            "pic": image_path,
            "emotion": emotion_label,
            "type": emotion_type, # 正确地基于 trial_selection 进行映射
            "correct_emotion": emotion_label
        })

        return results

    def find_image_pairs(self, experience_folder: str) -> List[Dict]:
        image_pairs = []
        # 修改正则表达式以匹配新的文件名格式
        experience_pattern = re.compile(r"\d+_\w+_\w+_(angry|disgusted|fearful|happy|sad|surprised)\.jpg$")

        for filename in os.listdir(experience_folder):
            match = experience_pattern.match(filename)
            if match:
                emotion = match.group(1)
                experience_path = os.path.join(experience_folder, filename)
                image_pairs.append({
                    "experience": experience_path,
                    "emotion": emotion
                })

        return image_pairs


def main():
    api_key = " "
    analyzer = EmotionAnalyzer(api_key)

    # 修改文件夹路径
    experience_folder = r"D:\PythonProject\T-LLM\sun-task\face-comb"

    image_pairs = analyzer.find_image_pairs(experience_folder)

    results = []
    for pair in tqdm(image_pairs, desc="Processing images"):
        result = analyzer.process_single_image(pair["experience"], pair["emotion"])
        if result:
            results.append(result)
        else:
            print(f"警告: 图像 {pair['experience']} 处理失败，可能无法识别情绪。但会继续处理其他信息。")

    df = pd.DataFrame(results)

    # 处理文件名
    df['pic'] = df['pic'].apply(lambda x: os.path.basename(x))

    # 重新排列列的顺序，与截图一致
    df = df[
        ['pic', 'emotion', 'textbox', 'valence', 'type', 'strength', 'correct_emotion', 'trial_selection', 'result']]

    # 保存结果
    excel_output_path = "combined_emotion_results31.xlsx" # 修改输出文件名
    df.to_excel(excel_output_path, index=False)
    print(f"表格已保存到: {excel_output_path}")


if __name__ == "__main__":
    main()