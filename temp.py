

import gradio as gr
import os
from google import genai
from google.genai import types
import PyPDF2
from moviepy import VideoFileClip
import json

# Gemini APIキーを環境変数から取得
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- 定数定義 ---
MODEL_NAME = "gemini-2.5-flash-lite" # 最新のモデルを利用

def analyze_material(file, current_text):
    """
    アップロードされたファイルを分析し、要約と抽出テキストを返す。
    """
    if file is None:
        return "分析するファイルをアップロードしてください。", current_text
    
    file_path = file.name
    _, file_extension = os.path.splitext(file_path)

    new_text = ""
    if file_extension.lower() == '.pdf':
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted_page_text = page.extract_text()
                    if extracted_page_text:
                        new_text += extracted_page_text + "\n"
        except Exception as e:
            return f"PDFの処理中にエラーが発生しました: {e}", current_text
    elif file_extension.lower() in ['.mp4', '.mov', '.avi']:
        new_text = "動画ファイルからのテキスト抽出は現在サポートされていません。"
        return f"`{os.path.basename(file_path)}`の分析: {new_text}", current_text
    else:
        return "サポートされていないファイル形式です。", current_text

    if not new_text.strip():
        return "ファイルからテキストを抽出できませんでした。", current_text

    updated_text = current_text + "\n" + new_text

    try:
        # Gemini APIを使用してテキストを要約
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"以下のテキストを詳細に要約してください:\n\n{updated_text}"
        )
        summary = response.text
        analysis_result = f"✅ `{os.path.basename(file_path)}` の分析が完了し、教材が追加されました。\n\n---\n\n**概要:**\n{summary}"
        return analysis_result, updated_text
    except Exception as e:
        return f"Gemini APIの呼び出し中にエラーが発生しました: {e}", updated_text

def chat_response(message, history, analyzed_text):
    """
    ユーザーの質問に対して、分析されたテキストをコンテキストとして回答を生成する。
    """
    if not analyzed_text or not analyzed_text.strip():
        return "まず「教材分析」タブで教材を分析してください。"

    try:
        system_prompt = f"""あなたは親切で優秀なアシスタントです。
        以下の『教材テキスト』に基づいて、ユーザーからの質問に忠実に、かつ分かりやすく答えてください。
        教材テキストに記載のない情報については、「教材に記載がありません」と正直に答えてください。

        ---教材テキスト---
        {analyzed_text}
        ---------------------"""

        gemini_history = []
        for user_msg, bot_msg in history:
            if user_msg:
                gemini_history.append({'role': 'user', 'parts': [{'text': user_msg}]})
            if bot_msg and "まず教材を分析してください" not in bot_msg:
                gemini_history.append({'role': 'model', 'parts': [{'text': bot_msg}]})
        
        # ユーザーの新しいメッセージを履歴に追加
        gemini_history.append({'role': 'user', 'parts': [{'text': message}]})

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=gemini_history,
            config=types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                top_k=64,
                max_output_tokens=8192,
                response_mime_type="text/plain",
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
            system_instruction=types.Content(parts=[types.Part(text=system_prompt)])
        )
        return response.text
    except Exception as e:
        return f"Gemini APIの呼び出し中にエラーが発生しました: {e}"

def generate_quiz_from_text(analyzed_text):
    if not analyzed_text or not analyzed_text.strip():
        raise gr.Error("クイズを生成するには、まず「教材分析」タブで教材を分析してください。")

    try:
        prompt = f"""以下の教材テキストに基づいて、内容の理解度を確認するための四択クイズを1問生成してください。
        回答の形式は、必ず以下のキーを持つJSONオブジェクトにしてください:
        - "question": (string) 問題文
        - "options": (list of strings) 4つの選択肢
        - "correct_answer": (string) 4つの選択肢のうち、正解の文字列
        - "explanation": (string) なぜその答えが正しいのかを、教材に基づいて説明する詳細な解説文

        ---教材テキスト---
        {analyzed_text}
        ---------------------"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            generation_config=types.GenerateContentConfig(
                temperature=0.8,
                top_p=0.95,
                top_k=64,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        
        quiz_data = json.loads(response.text)

        if not all(k in quiz_data for k in ['question', 'options', 'correct_answer', 'explanation']):
            raise gr.Error("クイズの生成に失敗しました。形式が正しくありません。")
        if len(quiz_data['options']) != 4:
            raise gr.Error("クイズの生成に失敗しました。選択肢が4つではありません。")

        return quiz_data['question'], quiz_data['options'], quiz_data['correct_answer'], quiz_data['explanation']

    except Exception as e:
        raise gr.Error(f"クイズの生成中にエラーが発生しました: {e}")

def check_answer(selected_option, correct_answer, explanation):
    if selected_option == correct_answer:
        result = f"### ✅ 正解です！素晴らしい！\n\n**解説:**\n{explanation}"
    else:
        result = f"### ❌ 不正解です。\n\n**正解:** {correct_answer}\n\n**解説:**\n{explanation}"
    return result

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# スマート教育アプリケーション")
    
    analyzed_text_state = gr.State("")

    with gr.Tabs():
        with gr.TabItem("教材分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 1. 教材のアップロード")
                    file_input = gr.File(label="PDFまたは動画ファイル（MP4, MOV, AVI）をアップロード")
                    analyze_button = gr.Button("分析して教材に追加", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("## 2. 分析結果")
                    analysis_output = gr.Markdown("ここに分析結果が表示されます。")
            analyze_button.click(
                analyze_material, 
                inputs=[file_input, analyzed_text_state], 
                outputs=[analysis_output, analyzed_text_state]
            )

        with gr.TabItem("インタラクティブQ&A"):
            gr.Markdown("## 教材に関する質問を入力してください")
            gr.ChatInterface(
                fn=chat_response, 
                chatbot=gr.Chatbot(height=400, label="チャットボット", avatar_images=('user.png', 'bot.png')),
                additional_inputs=[analyzed_text_state]
            )

        with gr.TabItem("理解度チェッククイズ"):
            gr.Markdown("## クイズに挑戦して理解度を確認しよう")
            
            generate_quiz_button = gr.Button("教材からクイズを生成", variant="primary")
            
            with gr.Group(visible=False) as quiz_container:
                quiz_question_display = gr.Markdown()
                quiz_options_radio = gr.Radio(label="選択肢")
                submit_answer_button = gr.Button("回答する")
                quiz_result_display = gr.Markdown()

            correct_answer_state = gr.State()
            explanation_state = gr.State()

            def show_quiz_ui(question, options, correct_answer, explanation):
                return {
                    quiz_container: gr.Group(visible=True),
                    quiz_question_display: gr.Markdown(f"**問題:** {question}"),
                    quiz_options_radio: gr.Radio(choices=options, value=None),
                    correct_answer_state: correct_answer,
                    explanation_state: explanation,
                    quiz_result_display: gr.Markdown()
                }

            generate_quiz_button.click(
                fn=generate_quiz_from_text,
                inputs=[analyzed_text_state],
                outputs=[quiz_question_display, quiz_options_radio, correct_answer_state, explanation_state]
            ).then(
                fn=show_quiz_ui,
                inputs=[quiz_question_display, quiz_options_radio, correct_answer_state, explanation_state],
                outputs=[quiz_container, quiz_question_display, quiz_options_radio, correct_answer_state, explanation_state, quiz_result_display]
            )

            submit_answer_button.click(
                fn=check_answer, 
                inputs=[quiz_options_radio, correct_answer_state, explanation_state], 
                outputs=quiz_result_display
            )

if __name__ == "__main__":
    if not os.path.exists("user.png"):
        from PIL import Image
        Image.new('RGB', (100, 100), color = 'gray').save('user.png')
    if not os.path.exists("bot.png"):
        from PIL import Image
        Image.new('RGB', (100, 100), color = 'blue').save('bot.png')
    demo.launch()
