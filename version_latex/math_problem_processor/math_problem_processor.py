
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sympy import symbols, Eq, solve
from gtts import gTTS
from manim import *
import moviepy.editor as mp
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# 1. Đọc và giảm kích thước dữ liệu từ file CSV
df = pd.read_csv('math_datasets.csv')
df_sample = df.sample(frac=0.1, random_state=42)  # Lấy mẫu 10% dữ liệu

# 2. Tiền xử lý dữ liệu văn bản
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.lower()  
    return text

df_sample['Input Text'] = df_sample['Input Text'].apply(clean_text)

# 3. Chuyển đổi văn bản thành vector số bằng TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df_sample['Input Text'])

# 4. Huấn luyện mô hình
X_train, X_test, y_train, y_test = train_test_split(X, df_sample['Math_Type'], test_size=0.2, random_state=42)

# Random Forest
model_rf = RandomForestClassifier(n_estimators=50, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print(f"Độ chính xác của Random Forest: {accuracy_score(y_test, y_pred_rf)}")

# Logistic Regression
model_lr = LogisticRegression(max_iter=100)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
print(f"Độ chính xác của Logistic Regression: {accuracy_score(y_test, y_pred_lr)}")

# 5. Giải quyết các bài toán toán học sử dụng SymPy
def solve_math_problem(problem_text):
    x = symbols('x')
    equation = Eq(x**2 + 5*x + 6, 0)
    solutions = solve(equation, x)
    return solutions

# Ví dụ giải bài toán
result = solve_math_problem("Solve the equation x^2 + 5x + 6 = 0")
print(result)

# 6. Chuyển đổi văn bản thành giọng nói và tạo video hoạt hình
def text_to_speech(text, output_file):
    tts = gTTS(text)
    tts.save(output_file)
    return output_file

audio_file = text_to_speech("The solutions are x equals minus 2 and x equals minus 3", "solution.mp3")

class MathAnimation(Scene):
    def __init__(self, equation_text, solution_text, **kwargs):
        self.equation_text = equation_text
        self.solution_text = solution_text
        super().__init__(**kwargs)

    def construct(self):
        equation = MathTex(self.equation_text)
        equation.to_edge(UP)
        self.play(Write(equation))

        solution = MathTex(self.solution_text)
        solution.next_to(equation, DOWN)
        self.play(Transform(equation, solution))
        self.wait(2)

def create_animation(math_solution, output_file):
    animation = MathAnimation("x^2 + 5x + 6 = 0", math_solution)
    animation.render()
    return output_file

video_file = create_animation("x = -2, x = -3", "solution.mp4")

def combine_audio_video(video_file, audio_file, output_file):
    video = mp.VideoFileClip(video_file)
    audio = mp.AudioFileClip(audio_file)
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_file, codec='libx264', audio_codec='aac')
    return output_file

final_video = combine_audio_video("solution.mp4", "solution.mp3", "final_output.mp4")

# 7. Xây dựng giao diện người dùng với Tkinter
audio_dir = "audio_files/"
video_dir = "video_files/"
final_dir = "final_videos/"

for directory in [audio_dir, video_dir, final_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_text():
    text = input_textbox.get("1.0", tk.END).strip()
    
    if not text:
        messagebox.showerror("Error", "Please enter a math problem text.")
        return

    try:
        math_solution = solve_math_problem(text)
        input_file_name = "problem"
        audio_file = text_to_speech(str(math_solution), audio_dir + input_file_name + '.mp3')
        video_file = create_animation(str(math_solution), video_dir + input_file_name + '.mp4')
        final_video = combine_audio_video(video_file, audio_file, final_dir + input_file_name + '_final.mp4')
        messagebox.showinfo("Success", f"Video created: {final_video}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

root = tk.Tk()
root.title("Math Problem to Video Converter")
tk.Label(root, text="Enter Math Problem:").pack(pady=5)
input_textbox = tk.Text(root, height=10, width=50)
input_textbox.pack(pady=5)
process_button = tk.Button(root, text="Convert to Video", command=process_text)
process_button.pack(pady=20)
root.mainloop()
