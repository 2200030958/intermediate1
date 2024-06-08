# inspect_data.py
def inspect_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
        for line in content[:5]:  # Display the first 5 lines for inspection
            print(line.strip())

if __name__ == "__main__":
    inspect_file('transfusion.data')
