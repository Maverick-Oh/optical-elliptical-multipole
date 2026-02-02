import time
import numpy as np
from typing import Union
import glob
import os
from astropy.wcs import WCS
import pandas
import warnings

def dict2str_newline(my_dict):
    my_string = ""
    for key, value in my_dict.items():
        if type(value) == np.ndarray: # a_m and phi_m
            if key == 'a_m':
                key = key+'x1000'
                value = np.array2string(value*1000, separator=', ', precision=3)
            elif key == 'phi_m':
                value = np.array2string(value, separator=', ', precision=3)
        my_string += f"{key}: {value:.3g}\n" if type(value) != str else f"{key}: {value}\n"
    return my_string

def grab_matching_format(data_dir, file_format):
    file_list = glob.glob(os.path.join(data_dir, file_format))
    if len(file_list) > 1:
        while True:
            for i, file in enumerate(file_list):
                print(f"{i}: {file}")
            play_sound_list(frequency_list=[220, 440, 880, 1760], duration_list=[0.5, 0.5, 0.5, 0.5])
            ind_selected = input("#### Which file to go with? Ansewr by putting an integer (e.g. 0, 1, 2): ")
            try:
                # Criteria to match
                cri1 = float(ind_selected) == int(ind_selected)
                cri2 = int(ind_selected) <= (len(file_list) - 1)
                cri3 = int(ind_selected) >= 0
                if cri1 and cri2 and cri3:
                    ind_selected = int(ind_selected)
                    file_target = file_list[ind_selected]
                    break
            except ValueError:
                print("Please enter a valid integer.")
            except Exception as e:
                raise e
    elif len(file_list) == 1:
        file_target = file_list[0]
        print("Folowing file chosen: ", file_target)
    else:
        raise ValueError(f"len(file_list)={len(file_list)}")
    return file_target


def elapsed_time_reporter(t0: float, i: int, total: int, seq_id: Union[int, str, None] = None, tail_str:str= ''):
    if type(seq_id) is str:
        seq_id = int(seq_id)
    done = i + 1
    elapsed = time.perf_counter() - t0
    # items/sec (avoid div by zero)
    rate = done / elapsed if elapsed > 0 else float('inf')
    rem = total - done
    eta_sec = rem / rate if np.isfinite(rate) and rate > 0 else float('nan')
    if np.isfinite(eta_sec):
        m, s = divmod(int(round(eta_sec)), 60)
        h, m = divmod(m, 60)
        eta_str = f"{h:02d}:{m:02d}:{s:02d}"
    else:
        eta_str = "--:--:--"
    msg = f"\r===== Processing: [{done:>5}/{total:<5}]  ETA: {eta_str}, sequentialid: {seq_id}, " + tail_str
    print(msg, end='', flush=True)
    if done == total:
        print("")
    return None

try:
    import pyaudio
except ImportError:
    pyaudio = None
    warnings.warn("pyaudio not found. Sound features will be disabled.")

def generate_sine_wave(fs, duration, frequency):
    samples = (np.sin(2 * np.pi * np.arange(fs * duration) * frequency / fs)).astype(np.float32)
    return samples

def play_sound(frequency = 880.0, duration = 2.0, volume = 0.5):
    if pyaudio is None:
        return None
    # Parameters for the sine wave
    fs = 44100  # sampling rate, Hz
    # Generate sine wave samples using numpy
    samples = generate_sine_wave(fs, duration, frequency)
    # Convert samples to bytes
    output_bytes = (volume * samples).tobytes()
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    # Open an audio stream
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    # Play the sound and then clean up the stream
    print(f"Playing sound for {duration:.2f} seconds at {frequency} Hz...")
    stream.write(output_bytes)
    print("Finished playing.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    return None

def play_sound_list(frequency_list = [220, 440, 880],
                    duration_list = [0.5, 0.5, 0.5], volume = 0.1, verbose = False):
    if pyaudio is None:
        return None
    # Parameters for the sine wave
    assert len(frequency_list) == len(duration_list)
    fs = 44100  # sampling rate, Hz
    # Generate sine wave samples using numpy
    samples = np.array([], dtype=np.float32)
    for freq, dur in zip(frequency_list, duration_list):
        sample = generate_sine_wave(fs, dur, freq)
        samples = np.concatenate((samples, sample))
    # Convert samples to bytes
    output_bytes = (volume * samples).tobytes()

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open an audio stream
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    # Play the sound and then clean up the stream
    if verbose:
        print(f"Playing sound for {duration_list} seconds at {frequency_list} Hz...")
    stream.write(output_bytes)
    if verbose:
        print("Finished playing.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    return None


def radec_to_pixel(hdu, ra, dec):
    w = WCS(hdu.header)
    x_pix, y_pix = w.world_to_pixel_values(ra, dec)
    return float(x_pix), float(y_pix)

def pixel_to_radec(hdu, x_pix, y_pix):
    """
    Convert pixel coordinates (x_pix, y_pix) to sky coordinates (RA, Dec)
    using the WCS in the FITS HDU header.

    Returns:
        ra (float), dec (float)   # in degrees
    """
    w = WCS(hdu.header)

    # Compute world coordinates (returns RA, Dec in degrees by default)
    sc = w.pixel_to_world(x_pix, y_pix)

    # Ensure output is float RA/Dec in degrees
    return float(sc.ra.deg), float(sc.dec.deg)

def upsert_sep_summary_row(row: dict, csv_path: str):
    """
    Insert or replace one SEP summary row into csv_path, using 'seqid' (string) as unique key.

    Behavior:
    - Creates the CSV if it does not exist.
    - Ensures seqid is stored as string.
    - Asserts at most one row per seqid in the existing CSV.
    - Takes the union of columns between existing CSV and the new row.
      Adds new columns (with NaN for old rows), and warns when schema expands.
    """
    import os

    # Normalize seqid to string
    if "seqid" not in row:
        raise KeyError("row must contain 'seqid' key")
    row = dict(row)  # shallow copy
    row["seqid"] = str(row["seqid"])

    if not os.path.exists(csv_path):
        # First time: just create the file
        df_new = pandas.DataFrame([row])
        df_new.to_csv(csv_path, index=False)
        return

    # Load existing file, enforcing seqid as string
    df = pandas.read_csv(csv_path, dtype={"seqid": str})
    if "seqid" not in df.columns:
        raise KeyError(f"'seqid' column not found in existing CSV: {csv_path}")

    # Check for duplicate seqid in existing file
    seq_col = df["seqid"].astype(str)
    counts = seq_col.value_counts()
    dup_seqids = counts[counts > 1]
    if len(dup_seqids) > 0:
        raise AssertionError(
            f"Found duplicate seqid(s) in {csv_path}: {list(dup_seqids.index)}"
        )

    # Build union of columns
    existing_cols = list(df.columns)
    row_cols = list(row.keys())
    # preserve order: existing first, then any new from row
    union_cols = list(dict.fromkeys(existing_cols + row_cols))

    # If new columns are being added, warn
    new_cols = [c for c in union_cols if c not in existing_cols]
    if new_cols:
        warnings.warn(
            f"upsert_sep_summary_row: adding new column(s) to {csv_path}: {new_cols}"
        )

    # Reindex existing DF to union of columns
    df = df.reindex(columns=union_cols)

    # Prepare the new row as a 1-row DataFrame with all union columns
    new_row_series = pandas.Series(row)
    new_row_series = new_row_series.reindex(union_cols)
    df_new_row = pandas.DataFrame([new_row_series])

    # Replace or append
    seqid_str = row["seqid"]
    mask = df["seqid"].astype(str) == seqid_str
    if mask.any():
        if mask.sum() > 1:
            raise AssertionError(
                f"Multiple rows with seqid={seqid_str} found in {csv_path}"
            )
        df.loc[mask, :] = df_new_row.iloc[0].values
    else:
        df = pandas.concat([df, df_new_row], ignore_index=True)

    # Save back
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    play_sound_list(frequency_list=[220,220,], duration_list=[0.5,0.5], volume=0.5)
    t0 = time.perf_counter()
    for i in range(5):
        elapsed_time_reporter(t0, i, 5)
    print("Done")