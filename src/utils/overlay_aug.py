import pandas as pd
import numpy as np
from pydub import AudioSegment
import os

np.random.seed(45)

dir = 'data/origin_sliced_train'
def overlay_audios(file_path1, file_path2, output_path):
    audio1 = AudioSegment.from_file(dir+file_path1[7:])
    audio2 = AudioSegment.from_file(dir+file_path2[7:])
    

    max_length = 5001
    # 랜덤 시작 위치 계산
    start_pos1 = np.random.randint(0, max(max_length - len(audio1), 0))
    start_pos2 = np.random.randint(0, max(max_length - len(audio2), 0))
    
    # 5초 길이의 빈 오디오 생성
    combined = AudioSegment.silent(duration=max_length)
    
    # 오디오 합성
    combined = combined.overlay(audio1, position=start_pos1)
    
    combined = combined.overlay(audio2, position=start_pos2)
    
    combined.export(output_path, format="ogg")

def generate_augmented_audios(csv_path, output_dir, num_real=0, num_fake=0, num_mixed=10000):
    df = pd.read_csv(csv_path)
    
    real_files = df[df['real'] == 1]['path'].tolist()
    fake_files = df[df['fake'] == 1]['path'].tolist()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def create_audio_pairs(file_list, num_pairs, used_indices):
        pairs = []
        for i in range(num_pairs):
            print(i, end=', ')
            while True:
                idx1, idx2 = np.random.choice(range(len(file_list)), 2, replace=False)
                if (idx1, idx2) not in used_indices and (idx2, idx1) not in used_indices:
                    used_indices.add((idx1, idx2))
                    break
            pairs.append((file_list[idx1], file_list[idx2]))
        return pairs

    used_real_indices = set()
    used_fake_indices = set()
    
    # print('real')
    # real_pairs = create_audio_pairs(real_files, num_real, used_real_indices)
    # print('fake')
    # fake_pairs = create_audio_pairs(fake_files, num_fake, used_fake_indices)

    
    mixed_pairs = []
    used_mixed_indices = set()
    
    while len(mixed_pairs) < num_mixed:
        print(len(mixed_pairs))
        while True:
            real_idx = np.random.choice(range(len(real_files)))
            fake_idx = np.random.choice(range(len(fake_files)))
            if (real_idx, fake_idx) not in used_mixed_indices:
                used_mixed_indices.add((real_idx, fake_idx))
                break
        used_mixed_indices.update([real_idx, fake_idx])
        mixed_pairs.append((real_files[real_idx], fake_files[fake_idx]))
    
    new_entries = []
    
    # for i, (file1, file2) in enumerate(real_pairs):
    #     output_path = os.path.join(output_dir, f"real_combined_{i+1}.ogg")
    #     overlay_audios(file1, file2, output_path)
    #     new_entries.append([f"real_combined_{i+1}", './train/'+f"real_combined_{i+1}"+'.ogg', 0, 1])
    #     print(f"real_combined_{i+1}")
    
    # for i, (file1, file2) in enumerate(fake_pairs):
    #     output_path = os.path.join(output_dir, f"fake_combined_{i+1}.ogg")
    #     overlay_audios(file1, file2, output_path)
    #     new_entries.append([f"fake_combined_{i+1}", './train/'+f"fake_combined_{i+1}"+'.ogg', 1, 0])
    #     print(f"fake_combined_{i+1}")
    
    for i, (file1, file2) in enumerate(mixed_pairs):
        output_path = os.path.join(output_dir, f"mixed_combined_{i+30000}.ogg")
        overlay_audios(file1, file2, output_path)
        new_entries.append([f"mixed_combined_{i+30000}", './train/'+f"mixed_combined_{i+30000}"+'.ogg', 1, 1])
        print(f"mixed_combined_{i+30000}")
    
    new_df = pd.DataFrame(new_entries, columns=['id', 'path', 'fake', 'real'])
    output_path = 'data/new_combined_train_10000.csv'
    new_df.to_csv(output_path, index=False)
    print(f"총 {len(new_entries)}개의 새로운 오디오 파일이 생성되고 CSV 파일에 추가되었습니다.")

if __name__ == "__main__":
    csv_path = "data/origin_sliced_train.csv"
    output_dir = "data/combined_train_10000"
    generate_augmented_audios(csv_path, output_dir)
