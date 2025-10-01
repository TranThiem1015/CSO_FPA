import numpy as np

def generate_new_h_array(K=4):
    """Generates a new KxK H array with complex values and saves it to file."""
    std = 1 / np.sqrt(2)  # For real/imag parts, E[|h|^2] = 1
    H = np.random.normal(0, std, (K, K)) + 1j * np.random.normal(0, std, (K, K))
    with open("hkl_array.txt", "w") as f:
        for row in H:
            f.write(" ".join([str(x) for x in row]) + "\n")
    return H

def read_h_array_from_txt_list(filename="hkl_array.txt"):
    """Reads a 2D complex H array from a text file."""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            if not lines:
                print(f"File {filename} is empty.")
                return None
            h_array = []
            for line in lines:
                # Remove brackets, commas, and extra whitespace
                line = line.strip().replace('[', '').replace(']', '').replace(',', '')
                if not line:  # Skip empty lines
                    continue
                row_str = line.split()
                try:
                    row_num = [complex(x) for x in row_str if x]
                    h_array.append(row_num)
                except ValueError as e:
                    print(f"Invalid data in line: {line}. Error: {e}")
                    return None
            if not h_array:
                print(f"No valid data found in {filename}.")
                return None
            # Ensure all rows have the same length
            if len(set(len(row) for row in h_array)) > 1:
                print(f"Inconsistent row lengths in {filename}.")
                return None
            return np.array(h_array, dtype=complex)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None

# Main logic (for testing)
if __name__ == "__main__":
    H = read_h_array_from_txt_list()
    if H is None:
        print("No valid H array found in file. Generating new values.")
        H = generate_new_h_array()
    print(f"Current H array:\n{H}")