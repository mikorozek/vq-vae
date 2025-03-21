import os
import glob
import librosa
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_and_process_audio(file_path, target_length=48000 * 4, sr=48000):
    data, orig_sr = librosa.load(file_path, sr=sr)
    
    if len(data) > target_length:
        max_offset = len(data) - target_length
        offset = np.random.randint(max_offset + 1)
        data = data[offset:(offset + target_length)]
    else:
        padded = np.zeros(target_length)
        padded[:len(data)] = data
        data = padded
    
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    
    return data

def get_all_audio_files(directory, extensions=["*.wav", "*.flac"]):
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return all_files

def split_dataset(all_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    train_files, test_files = train_test_split(
        all_files, 
        train_size=train_ratio,
        random_state=random_state
    )
    
    train_size = train_ratio / (val_ratio + train_ratio)
    train_files, val_files = train_test_split(
        train_files,
        train_size=train_size,
        random_state=random_state
    )
    
    return train_files, val_files, test_files

def save_dataset_hdf5(files, output_path, target_length=48000 * 4, 
                     sr=48000, chunk_size=100):
    total_files = len(files)
    print(f"Processing {total_files} files and saving to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as hf:
        audio_data = hf.create_dataset('audio_data', 
                                      shape=(total_files, target_length),
                                      dtype='float32',
                                      chunks=(chunk_size, target_length))
        
        dt = h5py.special_dtype(vlen=str)
        filenames = hf.create_dataset('filenames', 
                                     shape=(total_files,),
                                     dtype=dt)

        hf.attrs['total_files'] = total_files
        hf.attrs['sample_rate'] = sr
        hf.attrs['target_length'] = target_length
        
        for i, file_path in enumerate(tqdm(files, desc="Processing files")):
            try:
                audio = load_and_process_audio(file_path, target_length, sr)
                audio_data[i] = audio
                filenames[i] = os.path.basename(file_path)
                
                if (i + 1) % chunk_size == 0:
                    hf.flush()
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                audio_data[i] = np.zeros(target_length)
                filenames[i] = f"ERROR_{os.path.basename(file_path)}"
    
    print(f"Dataset saved to {output_path}")
    print(f"Approximate file size: {os.path.getsize(output_path) / (1024**3):.2f} GB")

def process_and_save_split_datasets(directory, output_dir, 
                                   train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                                   target_length=48000 * 4, sr=48000, chunk_size=100,
                                   extensions=["*.wav", "*.flac"], random_state=42):
    all_files = get_all_audio_files(directory, extensions)
    total_files = len(all_files)
    print(f"Found {total_files} audio files.")
    
    train_files, val_files, test_files = split_dataset(
        all_files, train_ratio, val_ratio, test_ratio, random_state
    )
    
    print(f"Split: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} test files")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.h5")
    val_path = os.path.join(output_dir, "val.h5")
    test_path = os.path.join(output_dir, "test.h5")
    
    save_dataset_hdf5(train_files, train_path, target_length, sr, chunk_size)
    save_dataset_hdf5(val_files, val_path, target_length, sr, chunk_size)
    save_dataset_hdf5(test_files, test_path, target_length, sr, chunk_size)
    
    print(f"All datasets saved to {output_dir}")
    return train_path, val_path, test_path

def load_dataset_batch(hdf5_path, batch_size=32, start_idx=0):
    with h5py.File(hdf5_path, 'r') as hf:
        total_files = hf.attrs['total_files']
        
        if start_idx >= total_files:
            return None, None  # End of dataset
        
        end_idx = min(start_idx + batch_size, total_files)
        actual_batch_size = end_idx - start_idx
        
        audio_batch = hf['audio_data'][start_idx:end_idx]
        filenames_batch = hf['filenames'][start_idx:end_idx]
        
        return audio_batch, filenames_batch


def dataset_generator(hdf5_path, batch_size=32):
    """Generator to yield batches from an HDF5 file"""
    with h5py.File(hdf5_path, 'r') as hf:
        total_files = hf.attrs['total_files']
        indices = np.arange(total_files)
        np.random.shuffle(indices)
        
        for start_idx in range(0, total_files, batch_size):
            end_idx = min(start_idx + batch_size, total_files)
            batch_indices = indices[start_idx:end_idx]
            
            audio_batch = np.array([hf['audio_data'][i] for i in batch_indices])
            filenames_batch = np.array([hf['filenames'][i] for i in batch_indices])
            
            yield audio_batch, filenames_batch

if __name__ == "__main__":
    train_path = "/home/mrozek/VCTK-processed/train.h5"

    audio_batch, filenames_batch = load_dataset_batch(train_path, batch_size=32)
    print(f"Loaded batch of {len(audio_batch)} samples")
    
    print("Example of batch generator usage:")
    for i, (audio_batch, filenames_batch) in enumerate(dataset_generator(train_path, batch_size=32)):
        print(f"Batch {i}: {len(audio_batch)} samples")
        if i >= 2:  
            break
