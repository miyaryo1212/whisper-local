import json
import os
import time
import tkinter.filedialog

import torch

import whisper


def select_file():
    file_types = [("Audio Files", "*.wav *.mp3 *.m4a *.ogg *.flac"), ("Video Files", "*.mp4 *.mkv *.mov *.avi *.flv"), ("All Files", "*.*")]
    file_path = tkinter.filedialog.askopenfilename(title="Select an audio or video file", filetypes=file_types)
    return file_path


def transcribe_file(file_path):
    # GPUが利用可能かどうかの確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Whisperモデルのロード
    model_name = "large"  # tiny, base, small, medium, large, large-v2, large-v3
    model = whisper.load_model(model_name, device=device)

    # タイマー計測開始
    t0 = time.perf_counter()

    # 音声ファイルの文字起こし
    result = model.transcribe(file_path)

    # タイマー計測終了
    t1 = time.perf_counter()
    print(f"Transcribed in {t1-t0:.02f}s")

    # 文字起こし結果の出力
    print(result["text"])

    # 保存先のフォルダ
    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)

    # ファイル名
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # 結果をJSONファイルに保存
    json_output_path = f"./output/{file_name}_transcription_{model_name}.json"
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

    # テキスト部分をTXTファイルに保存
    txt_output_path = f"./output/{file_name}_transcription_{model_name}.txt"
    with open(txt_output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(str(result["text"]))

    print(f"Transcription result saved as {json_output_path} and {txt_output_path}")


if __name__ == "__main__":
    file_path = select_file()
    if file_path:
        transcribe_file(file_path)
    else:
        print("No file selected")
