# Sudoku-OCR

Smart Sudoku Number Detection that tries to extract a sudoku board from a photo and detect number from it.

## Table Of Contents:

[Installation](https://github.com/neeru1207/AI_Sudoku/blob/master/README.md#installation)

[Usage](https://github.com/neeru1207/AI_Sudoku/blob/master/README.md#usage)

[Working](https://github.com/neeru1207/AI_Sudoku/blob/master/README.md#working)

- [Image Preprocessing](https://github.com/neeru1207/AI_Sudoku/blob/master/README.md#image-preprocessing)
- [Recognition](https://github.com/neeru1207/AI_Sudoku/blob/master/README.md#recognition)

## Installation

### 1. Download and install Python3 from [here](https://www.python.org/downloads/)

Note: You can install one of the following methods below:

### Method 1: Installation with virtualenv (Recommended)

1. I recommend using [virtualenv](https://virtualenv.pypa.io/en/latest/). Download virtualenv by opening a terminal and typing:
   ```bash
   pip install virtualenv
   ```
2. Create a virtual environment with the name sudokuenv.

   - Windows

   ```bash
   virtualenv sudokuenv
   cd sudokuenv/Scripts
   activate
   ```

   - Linux:

   ```bash
   source sudokuenv/bin/activate
   ```

3. Clone this repository, extract it if you downloaded a .zip or .tar file and cd into the cloned repository.

   - For Example:

   ```bash
   cd A:\SudokuOCR
   ```

4. Install the required packages by typing:
   ```bash
   pip install -r requirements.txt
   ```
5. Run board.py file to crop board image:
   ```bash
   python3 board.py
   ```
6. Run run.py file to detected number from image:
   ```bash
   python3 run.py
   ```

### Method 2: Run directly from root directory

1. Run board.py file to crop board image:
   ```bash
   python3 board.py
   ```
2. Run run.py file to detected number from image:
   ```bash
   python3 run.py
   ```
