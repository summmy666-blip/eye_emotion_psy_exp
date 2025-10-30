import os
import re
import time
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Optional
import pandas as pd
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

class EmotionAnalyzer:
    def __init__(self, num_workers=1):
        # Qwen 模型配置
        torch.cuda.empty_cache()  # 释放显存缓存

        print("正在加载Qwen模型...")
        model_id = "/app/data/huggingface_models/Qwen2.5-VL-72B-Instruct"

        # 设备检测
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"检测到设备: {self.device}")

        # 加载模型
        try:
            print("尝试使用torch_dtype加载模型...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16  # 使用fp16替代bfloat16
            ).eval()
        except Exception as e:
            print(f"使用torch_dtype加载失败: {e}")
            print("尝试使用默认配置加载模型...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                device_map="auto"
            ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"模型加载完成，使用设备: {self.device}")
        
        # 情绪类型
        self.emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]

        # 情绪类型到编号的映射
        self.emotion_to_number = {
            "angry": 1,
            "disgusted": 2,
            "fearful": 3,
            "happy": 4,
            "sad": 5,
            "surprised": 6,
            "neutral": 7
        }

        # 预置提示词
        self.prompts = {
            "description": """
            这组图片包含两张面孔/两双眼睛。左图展示的是中性情绪，右图展示的是同一个人表达的六个基本情绪之一。请参考左图，分析右图表达的是什么眼神。
            **重要提示：**明确用两个字的形容词表述，不要有多余的话，另外不要回答无法判断，如果判断不了就回答一个可能性大的答案。
            """,

            "emotion_valence": """
            这组图片包含两张面孔/两双眼睛。左图展示的是中性情绪，右图展示的是同一个人表达的六个基本情绪之一。请参考左图，分析右图传达的情绪属于什么效价。
            **重要提示：**你的答案必须是一个介于1.00和5.00之间的数字，精确到小数点后2位。
            例如：2.75、3.90、4.25等。除非你认为必要，否则不要使用.00作为小数点。
            这些数字表示从非常积极（1）到较为积极（2）到中性（3）到较为消极（4）到非常消极（5）的范围。 
            只提供数字，没有其他文字或解释。
            """,

            "emotion_type": """
            这组图片包含两张面孔/两双眼睛。左图展示的是中性情绪，右图展示的是同一个人表达的六个基本情绪之一。请参考左图，分析右图传达的情绪类型是什么。
            **重要提示：** 你必须从以下词语中选择一个且仅一个作为你的答案：
            angry, disgusted, fearful, happy, sad, surprised
            不要回答其他任何词语！不要说"unknown"。不要提供任何解释。
            """,

            "emotion_intensity": """
            这组图片包含两张面孔/两双眼睛。左图展示的是中性情绪，右图展示的是同一个人表达的六个基本情绪之一。请参考左图，分析右图传达的情绪有多强。
            **重要提示：**你的答案必须是一个介于1.00和5.00之间的数字，精确到小数点后2位。
            例如：2.75、3.90、4.25等。除非你认为必要，否则不要使用.00作为小数点。
            这些数字代表了从非常弱（1）到非常强（5）的等级。
            只提供数字，没有其他文字或解释。
            """
        }

        # 并行处理配置
        self.num_workers = num_workers

        # 设置请求重试参数
        self.max_retries = 6
        self.base_delay = 1
        self.max_delay = 10
        self.timeout = 60

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载图像文件为PIL Image对象"""
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                return image
            else:
                print(f"图像文件不存在: {image_path}")
                return None
        except Exception as e:
            print(f"加载图像文件失败: {image_path}, 错误: {e}")
            return None

    def api_request(self, prompt: str, combined_image_path: str, prompt_type: str, subject_id: str) -> Optional[str]:
        """使用Qwen模型进行本地推理，处理拼图图像"""
        # 加载图像
        combined_image = self.load_image(combined_image_path)
        
        if combined_image is None:
            return None

        # 构建消息
        messages = []
        
        # 添加系统消息
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": "你是一个专业的情绪分析助手，根据提供的拼图图片进行分析。请保持简洁明了的回答。"}]
        })

        # 添加当前问题
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": combined_image},
                {"type": "text", "text": prompt}
            ],
        })

        print(f"正在处理图像: {subject_id} - {os.path.basename(combined_image_path)}")
        print(f"消息数量: {len(messages)}")
        print(f"提示词类型: {prompt_type}")

        for attempt in range(self.max_retries):
            try:
                # 应用聊天模板
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device)

                # 记录输入长度
                input_len = inputs["input_ids"].shape[-1]

                # 生成回应
                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=150,  
                        temperature=1,
                    )
                    # 只提取新生成的部分
                    generation = generation[0][input_len:]

                # 解码回应
                response = self.processor.decode(generation, skip_special_tokens=True)
                print(f"模型原始回复: {response[:100]}...")

                # 提取答案
                extracted_answer = self.extract_answer(response, prompt_type)
                print(f"提取的答案: {extracted_answer}")
                
                return extracted_answer

            except Exception as e:
                print(f"模型推理异常 (第 {attempt + 1} 次尝试): {e}")
                import traceback
                traceback.print_exc()  # 打印详细错误信息

                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
                else:
                    print("达到最大重试次数，放弃处理")
                    return None

        return None  # 所有重试都失败

    def extract_answer(self, response_text, prompt_type):
        """从响应文本中提取答案"""
        text = response_text.strip()

        if prompt_type == "emotion_type":
            # 提取情绪类型 (angry, disgusted, fearful, happy, sad, surprised)
            for emotion in ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]:
                if emotion in text.lower():
                    return emotion
            return "unknown"

        elif prompt_type in ["emotion_valence", "emotion_intensity"]:
            # 提取数值 (1.00-5.00)
            pattern = re.compile(r'(\d+\.\d{2}|\d+)')
            match = pattern.search(text)
            if match:
                value = float(match.group(1))
                if 1 <= value <= 5:
                    return f"{value:.2f}"
            return "3.00"  # 默认中性值

        elif prompt_type == "description":
            # 提取两个字的中文形容词
            pattern = re.compile(r'[\u4e00-\u9fff]{2}')
            match = pattern.search(text)
            if match:
                return match.group(0)
            return "未知"

        return text

    def find_emotion_images(self, combined_folder):
        """查找合并后的拼图图像文件"""
        image_files = {}

        try:
            # 列出文件夹中的所有文件
            all_files = os.listdir(combined_folder)
            print(f"找到 {len(all_files)} 个拼图图像文件")

            # 过滤出图像文件
            image_files_list = [f for f in all_files if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # 对每个图像文件提取信息
            for filename in image_files_list:
                # 提取ID和情绪
                parts = filename.split('_')
                if len(parts) < 3:  # 确保有足够的部分，如 12_Caucasian_female_angry.jpg
                    continue

                # ID在第一个位置
                subject_id = parts[0]

                # 情绪在文件名的最后部分，去掉扩展名
                last_part = parts[-1].split('.')[0].lower()
                if last_part in self.emotions and last_part != "neutral":
                    # 添加到图像文件字典
                    if subject_id not in image_files:
                        image_files[subject_id] = {}
                    
                    image_files[subject_id][last_part] = os.path.join(combined_folder, filename)

            print(f"找到 {len(image_files)} 个有效主体，包含情绪图像拼图")
            return image_files

        except Exception as e:
            print(f"查找图像文件时出错: {e}")
            return {}

    def analyze_image(self, subject_id, emotion_path, emotion_type):
        """分析单张拼图图像并返回结果"""
        print(f"分析主体 {subject_id} 的 {emotion_type} 情绪...")

        # 分析结果
        results = {
            "subject_id": subject_id,
            "emotion_actual": emotion_type,
            "emotion_image": os.path.basename(emotion_path)
        }

        # 对每种提示进行分析
        for prompt_type, prompt in self.prompts.items():
            try:
                answer = self.api_request(prompt, emotion_path, prompt_type, subject_id)
                results[prompt_type] = answer if answer else "unknown"
            except Exception as e:
                print(f"分析 {prompt_type} 时出错: {e}")
                results[prompt_type] = "error"

        return results

    def process_all_images(self, image_files, output_folder, file_num):
        """处理所有拼图图像并实时保存结果"""
        all_results = []

        # 计算总任务数
        total_tasks = sum(len(emotions) for emotions in image_files.values())

        # 创建进度条
        print(f"总共需要处理 {total_tasks} 个拼图图像")
        pbar = tqdm(total=total_tasks, desc="处理进度")

        # 创建临时结果文件
        temp_results_file = os.path.join(output_folder, f"temp_results_{file_num:02d}.xlsx")

        # 创建一个列顺序
        columns_order = ['pic', 'emotion', 'textbox', 'valence', 'type', 'strength', 'correct_emotion',
                         'trial_selection', 'result']

        # 如果已经有临时文件，先读取
        existing_results = []
        if os.path.exists(temp_results_file):
            try:
                existing_df = pd.read_excel(temp_results_file)
                existing_results = existing_df.to_dict('records')
                print(f"读取到 {len(existing_results)} 条已存在的结果")
                # 更新进度条
                pbar.update(len(existing_results))
                all_results = existing_results
            except Exception as e:
                print(f"读取临时文件时出错: {e}")

        # 用于记录已处理的图片，避免重复处理
        processed_images = set()
        for result in all_results:
            processed_images.add(result.get('pic', ''))

        for subject_id, emotions in image_files.items():
            # 对每种情绪图片进行分析
            for emotion, path in emotions.items():
                # 检查是否已处理过该图片
                pic_name = os.path.basename(path)

                if pic_name in processed_images:
                    print(f"跳过已处理的图片: {pic_name}")
                    pbar.update(1)
                    continue

                try:
                    # 分析图片
                    result = self.analyze_image(subject_id, path, emotion)
                    if result:
                        # 格式化单条结果
                        formatted_result = self.format_single_result(result)
                        all_results.append(formatted_result)
                        processed_images.add(formatted_result['pic'])

                        # 实时保存结果
                        temp_df = pd.DataFrame(all_results)
                        # 确保列的顺序
                        for col in columns_order:
                            if col not in temp_df.columns:
                                temp_df[col] = ""
                        temp_df = temp_df[columns_order]
                        temp_df.to_excel(temp_results_file, index=False)
                        print(f"\n已保存临时结果，当前处理了 {len(all_results)} 条数据")

                except Exception as e:
                    print(f"处理主体 {subject_id} 的 {emotion} 图片时出错: {e}")
                finally:
                    pbar.update(1)  # 无论成功失败都更新进度条

        pbar.close()
        return all_results

    def format_single_result(self, result):
        """格式化单条结果以符合指定的输出格式"""
        # 提取情绪图片名称
        pic = result['emotion_image']

        # 映射字段到新格式 - 使用预测的emotion_type
        formatted_result = {
            'pic': pic,
            'emotion': result['emotion_actual'],
            'textbox': result['description'],
            'valence': result['emotion_valence'],
            'type': self.emotion_to_number.get(result['emotion_type'], 0),  # 使用预测值而非真实值
            'strength': result['emotion_intensity'],
            'correct_emotion': result['emotion_actual'],
            'trial_selection': result['emotion_type'],
            'result': 1 if result['emotion_actual'] == result['emotion_type'] else 0
        }

        return formatted_result

def main():
    # 修改为拼图图像的路径
    combined_folder = "/home/majc_docker/sun/face-comb/"

    # 创建结果文件夹
    output_folder = os.path.join(os.path.dirname(combined_folder), "results")
    os.makedirs(output_folder, exist_ok=True)

    # 创建分析器
    analyzer = EmotionAnalyzer(num_workers=1)

    # 查找图像文件
    print("正在查找拼图图像文件...")
    image_files = analyzer.find_emotion_images(combined_folder)

    if not image_files:
        print("没有找到有效的拼图图像文件，程序结束")
        return

    # 生成结果文件
    for file_num in range(2, 37):  # 可以调整范围
        # 记录开始时间
        start_time = time.time()
        print(f"\n开始生成第 {file_num} 个结果文件")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 处理所有图片，使用实时保存方法
        print("开始分析图片...")
        results = analyzer.process_all_images(image_files, output_folder, file_num)

        if not results:
            print("分析结果为空")
            continue

        # 保存最终结果到Excel，使用编号命名
        output_file = os.path.join(output_folder, f"emotion_results_{file_num:02d}.xlsx")
        df = pd.DataFrame(results)

        # 确保列的顺序符合要求
        columns_order = ['pic', 'emotion', 'textbox', 'valence', 'type', 'strength', 'correct_emotion',
                         'trial_selection', 'result']
        for col in columns_order:
            if col not in df.columns:
                df[col] = ""
        df = df[columns_order]

        df.to_excel(output_file, index=False)

        # 删除临时文件
        temp_file = os.path.join(output_folder, f"temp_results_{file_num:02d}.xlsx")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"临时文件已删除: {temp_file}")
            except Exception as e:
                print(f"删除临时文件时出错: {e}")

        # 计算总用时
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"第 {file_num} 个文件生成完成，共处理 {len(results)} 个图片")
        print(f"总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        print(f"结果已保存到: {output_file}")

    print("\n所有文件生成完成！")

if __name__ == "__main__":
    main()