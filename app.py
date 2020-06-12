import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import StringIO, BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib.patches
from PuzzleExtractor.digit_extraction import extractDigits
from PuzzleExtractor.processing import processImage
from PuzzleExtractor.grid_extraction import extractGrid
import GridSolver.SudokuSolve as Solver
from DigitsRecogniser.recognise_digits import predictDigits

def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

def PIL_image_to_bytes(img_obj: Image.Image):
    buf = BytesIO()
    img_obj.save(buf, format='JPEG')
    img_bytes = buf.getvalue()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def show_image_with_pyplot(img, caption, grayscale=True, ax=None):
    if(len(img.shape) == 3): img_plt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else                   : img_plt = img

    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes()

    ax.set_title(caption)
    ax.set_xticks([])
    ax.set_yticks([])

    if grayscale:   ax.imshow(img_plt, cmap='gray', vmin=0, vmax=255)
    else:           ax.imshow(img_plt, vmin=0, vmax=255)


# Plot's sudoku passed as numpy array (n) on pyplot axis (ax)
def plot_sudoku(n: np.ndarray, caption:str, ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(caption)

    # Simple plotting statement that ingests a 9x9 array (n), and plots a sudoku-style grid around it.
    for y in range(10):
        ax.plot([-0.05,9.05],[y,y],color='black',linewidth=1)

    for y in range(0,10,3):
        ax.plot([-0.05,9.05],[y,y],color='black',linewidth=3)

    for x in range(10):
        ax.plot([x,x],[-0.05,9.05],color='black',linewidth=1)

    for x in range(0,10,3):
        ax.plot([x,x],[-0.05,9.05],color='black',linewidth=3)

    # plt.axis('image')
    # plt.axis('off') # drop the axes, they're not important here

    for x in range(9):
        for y in range(9):
            foo=n[8-y][x] # need to reverse the y-direction for plotting
            if foo > 0: # ignore the zeros
                T=str(foo)
                ax.text(x+0.3,y+0.2,T,fontsize=20)


# Assumes digits in column major order
def get_extracted_digits_img(digits: list, rows: int, cols: int) -> np.ndarray:
  combined = np.array([])
  for i in range(cols):
    bordered_digit = cv2.copyMakeBorder(digits[i*rows],2,2,2,2,cv2.BORDER_CONSTANT,value=255)
    col_combined = bordered_digit
    for j in range(1, rows):
      bordered_digit = cv2.copyMakeBorder(digits[i*rows + j],2,2,2,2,cv2.BORDER_CONSTANT,value=255)
      col_combined = np.vstack((col_combined, bordered_digit))

    if i == 0:
      combined = col_combined
    else:
      combined = np.hstack((combined, col_combined))

  return combined


def get_np_array_row_major(inp: str, rows: int, cols: int) -> np.ndarray:
  inp_array = np.fromstring(inp, dtype=np.int8, sep='') - 48
  return np.reshape(inp_array, (rows, cols))


def get_patches(ax1, ax2, ax3, ax4):
  limits_x = [axis.get_xlim() for axis in [ax1, ax2, ax3, ax4]]
  limits_y = [axis.get_ylim() for axis in [ax1, ax2, ax3, ax4]]
  patch_type = "-|>"

  patch1 = matplotlib.patches.ConnectionPatch(
    xyA=((limits_x[0][1]+limits_x[0][0])//2, limits_y[0][0]),
    xyB=((limits_x[1][1]+limits_x[1][0])//2, limits_y[1][1]),
    coordsA="data",
    coordsB="data",
    axesA=ax1,
    axesB=ax2,
    arrowstyle=patch_type,
    color="green",
    shrinkA=10,
    mutation_scale=60,
    linewidth=10,
    alpha=0.8,
    clip_on=False,
  )

  patch2 = matplotlib.patches.ConnectionPatch(
    xyA=(limits_x[1][1], (limits_y[1][0]+limits_y[1][1])//2),
    xyB=(limits_x[2][0], (limits_y[2][0]+limits_y[2][1])//2),
    coordsA="data",
    coordsB="data",
    axesA=ax2,
    axesB=ax3,
    arrowstyle=patch_type,
    color="green",
    shrinkA=10,
    mutation_scale=60,
    linewidth=10,
    alpha=0.8,
    clip_on=False,
  )

  patch3 = matplotlib.patches.ConnectionPatch(
    xyA=((limits_x[2][0]+limits_x[2][1])//2, limits_y[2][1]),
    xyB=((limits_x[3][0]+limits_x[3][1])//2, limits_y[3][0]),
    coordsA="data",
    coordsB="data",
    axesA=ax3,
    axesB=ax4,
    arrowstyle=patch_type,
    color="green",
    shrinkA=10,
    mutation_scale=60,
    linewidth=10,
    alpha=0.8,
    clip_on=False,
  )

  return (patch1, patch2, patch3)



def get_matplotlib_figure(orig_img_array: np.ndarray, extracted_digits: np.ndarray, recognised_digits: np.ndarray, solved_grid: np.ndarray) -> plt.figure:
  fig, axes = plt.subplots(2,2)
  fig.set_size_inches(15,15)

  # Plotting original image
  caption="Original Image"
  show_image_with_pyplot(orig_img_array, caption, grayscale=False, ax=axes[0,0])

  # Plotting extracted digits
  caption="Extracted digits from Image"
  digits_img = get_extracted_digits_img(extracted_digits, 9, 9)
  show_image_with_pyplot(digits_img, caption, grayscale=True, ax=axes[1, 0])

  # Plotting recognised digits
  caption="Recognised digits using the selected model"
  plot_sudoku(recognised_digits, caption, axes[1,1])

  if solved_grid is not None:
    # Plotting solved_grid
    fig.suptitle("Successfully solved the sudoku")
    caption = "Solved sudoku using Backtracking"
    plot_sudoku(solved_grid, caption, axes[0,1])
  else:
      fig.suptitle("Error !!! The sudoku recognised is invalid and cannot be solved.")

  patches1, patches2, patches3 = get_patches(axes[0,0], axes[1,0], axes[1,1], axes[0,1])
  axes[0,0].add_artist(patches1)
  axes[1,0].add_artist(patches2)
  axes[1,1].add_artist(patches3)

  return fig


#def display_PIL_img(img_obj: Image.Image, caption: str, width_str: str):
#  st.write(caption)
#  header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(PIL_image_to_bytes(img_obj))
#  st.markdown(header_html, unsafe_allow_html=True)


def str_to_np_arr(inp_str):
  try:
    res = np.zeros((9,9), dtype=np.int8)
    rows = inp_str[1:-1].split('\n ')
    #print(rows)
    for i in range(9):
      curr_row = rows[i][1:-1].split(' ')
      for j in range(9):
        res[i][j] = int(curr_row[j])
        if(res[i][j] > 9 or res[i][j] < 0):
            return None
    
    return res
  except:
    return None


def operate_on_image(img_array, model):
  binary_img_array = processImage(img_array)
  status, grid_array = extractGrid(binary_img_array)
  if status == False:
    st.write("The Sudoku grid cannot be extracted from the image")
  else:
    extracted_digits = extractDigits(grid_array)
    recognised_digits_str = predictDigits(extracted_digits, model)
    recognised_digits = get_np_array_row_major(recognised_digits_str, 9, 9)
    grid_text = st.sidebar.text_area("Modify the parsed sudoku grid to correct the parser", value=recognised_digits, max_chars=200)

    if grid_text != str(recognised_digits):
      grid_text_arr = str_to_np_arr(grid_text)
      if grid_text_arr is not None:
        recognised_digits = grid_text_arr
        recognised_digits_str = ''.join(c for c in grid_text if c not in '[]\n ')
      else:
        st.sidebar.markdown("Invalid grid entered")

    solver = Solver.Sudoku(recognised_digits_str)

    if((not solver.verifyGridStatus()) or (not solver.solveGrid())):
        st.write("The puzzle extracted is invalid")
        figure = get_matplotlib_figure(img_array, extracted_digits, recognised_digits, None)
    else:
        solved_grid_str = solver.getGrid()
        solved_grid = get_np_array_row_major(solved_grid_str, 9, 9)
        figure = get_matplotlib_figure(img_array, extracted_digits, recognised_digits, solved_grid)

    st.pyplot(figure)



@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def read_example_image(name):
  img = cv2.imread("ExampleImages/" +  name + ".jpg")
  return img


_max_width_()

st.title("Sudoku Image Solver")
st.sidebar.markdown("<style> .sidebar-content, .block-container{ margin: 0;padding:0rem 0rem 0rem 0.5rem !important; font-family: Arial, Helvetica, sans-serif;} textarea{min-height:250px !important;}</style> <h2> Sudoku Image Solver </h2>", unsafe_allow_html=True)

"""
---  

Tools and libraries used:   
OpenCV-Python       : For using image processing algorthims for Grid Extraction and Digits Extraction.  
Keras, Scikit-Learn : For implementing ML models for recognizing digits extracted from the image  
I have also used Swig for creating Python bindings of C++ code for solving the puzzle recognized from the image.  

Author: [Vaibhav Thakkar](https://github.com/vaithak)   
You can find the code [here](https://github.com/vaithak/SudokuImageSolver)  

---  
"""


images_list = ["image" + str(i) for i in range(1,8)]
models = ["Convolutional Neural Networks", "XGBoost Algorithm", "Softmax Regression"]


option = st.sidebar.radio('Please choose any of the following options',('Choose image from examples','Upload your own image'))
model = st.sidebar.radio("Model for recognizing digits", models, 0)

input_image = None
if option == "Choose image from examples":
    example_image_name = st.sidebar.selectbox("Select an example image", images_list)
    input_image = read_example_image(example_image_name)
else:
    uploaded_file = st.file_uploader("Upload your own image (supported types: jpg, jpeg, png)...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        input_image = np.array(img)
        input_image = input_image[:,:,::-1]


if input_image is not None:
    operate_on_image(input_image, models.index(model))

