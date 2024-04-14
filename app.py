# import the frameworks, packages and libraries 
import streamlit as st 
from PIL import Image 
from io import BytesIO 
import numpy as np 
import cv2 # computer vision 



def convertto_watercolorsketch(inp_img): 
	img_1 = cv2.edgePreservingFilter(inp_img, flags=2, sigma_s=2.5, sigma_r=0.95328428) 
	img_water_color = cv2.stylization(img_1, sigma_s=2.5, sigma_r=0.15218328425) 
	return(img_water_color) 

# function to convert an image to a pencil sketch 

def pencilsketch(inp_img): 
	img_pencil_sketch, pencil_color_sketch = cv2.pencilSketch( 
		inp_img, sigma_s=1.0, sigma_r=0.2, shade_factor=0.191418) 
	return(img_pencil_sketch) 


def apply_emboss_effect(image):
    # Define the kernel for embossing
    kernel_emboss = np.array([[0, -1, -1],
                              [1,  0, -1],
                              [1,  1,  0]])
    
    # Apply the filter to the image
    embossed_image = cv2.filter2D(image, -1, kernel_emboss)
    
    # Normalize the result to keep it in the [0, 255] range
    embossed_image = cv2.normalize(embossed_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return embossed_image

def apply_emboss_effect(image):
    # Define the kernel for embossing
    kernel_emboss = np.array([[0.901, 0.01, -1],
                              [1,  -0.1, -1],
                              [1,  1,  0]])
    
    # Apply the filter to the image
    embossed_image = cv2.filter2D(image, -1, kernel_emboss)
    
    # Normalize the result to keep it in the [0, 255] range
    embossed_image = cv2.normalize(embossed_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return embossed_image

# function to apply emboss effect and display the result
def apply_and_display_emboss_effect(image_file):
    # Load the image
    image = cv2.imread(image_file)
    
    # Apply emboss effect
    embossed_image = apply_emboss_effect(image)
    
    # Convert NumPy array to PIL image for display in Streamlit
    embossed_pil = Image.fromarray(embossed_image)
    
    # Display the original and embossed images
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original Image")
        st.image(load_an_image(image_file), width=250)
    
    with col2:
        st.header("Emboss Effect")
        st.image(embossed_pil, width=250)

def apply_cartoon_filter(image, num_down=2, num_bilateral=7):
    # Downsample the image using Gaussian Pyramid
    img_color = image
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)

    # Apply bilateral filter multiple times
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

    # Upsample the image to its original size
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply median blur to reduce noise
    img_blur = cv2.medianBlur(img_gray, 7)

    # Detect edges using adaptive thresholding
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

    # Combine the color image with the edges
    img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edge)

    return img_cartoon


# function to load an image 
def load_an_image(image): 
	img = Image.open(image) 
	return img 


# the main function which has the code for 
# the web application 
def main(): 
	
	
	# basic heading and titles 
	st.title('WEB APPLICATION TO CONVERT IMAGE TO SKETCH') 
	st.write("This is an application developed for converting\ your ***image*** to a ***Water Color Sketch*** OR ***Pencil Sketch***") 
	st.subheader("Please Upload your image")
	# image file uploader 
	image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"]) 

	# if the image is uploaded then execute these 
	# lines of code 
	if image_file is not None: 
		
		# select box (drop down to choose between water 
		# color / pencil sketch) 
		option = st.selectbox('How would you like to convert the image', 
							('Convert to water color sketch', 
							'Convert to pencil sketch','Apply emboss effect', 'Apply cartoon filter' )) 
		if option == 'Convert to water color sketch': 
			image = Image.open(image_file) 
			final_sketch = convertto_watercolorsketch(np.array(image)) 
			im_pil = Image.fromarray(final_sketch) 

			# two columns to display the original image and the 
			# image after applying water color sketching effect 
			col1, col2 = st.columns(2) 
			with col1: 
				st.header("Original Image") 
				st.image(load_an_image(image_file), width=250) 

			with col2: 
				st.header("Water Color Sketch") 
				st.image(im_pil, width=250) 
				buf = BytesIO() 
				img = im_pil 
				img.save(buf, format="JPEG") 
				byte_im = buf.getvalue() 
				st.download_button( 
					label="Download image", 
					data=byte_im, 
					file_name="watercolorsketch.png", 
					mime="image/png"
				) 

		if option == 'Convert to pencil sketch': 
			image = Image.open(image_file) 
			final_sketch = pencilsketch(np.array(image)) 
			im_pil = Image.fromarray(final_sketch) 
			
			# two columns to display the original image 
			# and the image after applying 
			# pencil sketching effect 
			col1, col2 = st.columns(2) 
			with col1: 
				st.header("Original Image") 
				st.image(load_an_image(image_file), width=250) 

			with col2: 
				st.header("Pencil Sketch") 
				st.image(im_pil, width=250) 
				buf = BytesIO() 
				img = im_pil 
				img.save(buf, format="JPEG") 
				byte_im = buf.getvalue() 
				st.download_button( 
					label="Download image", 
					data=byte_im, 
					file_name="watercolorsketch.png", 
					mime="image/png")
		if option == 'Apply emboss effect':  # New option selected
			image = Image.open(image_file)
			img_array = np.array(image)
			embossed_image = apply_emboss_effect(img_array)  # Apply emboss effect
			embossed_pil = Image.fromarray(embossed_image)  # Convert NumPy array to PIL image
			col1, col2 = st.columns(2)
			with col1:
				st.header("Original Image")
				st.image(load_an_image(image_file), width=250)
			with col2:
				st.header("Emboss Effect")
				st.image(embossed_pil, width=250)
				buf = BytesIO()
				img = embossed_pil
				img.save(buf, format="JPEG")
				byte_im = buf.getvalue()
				st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="embossed_image.png",  # Change file name for embossed image
                    mime="image/png")
		if option == 'Apply cartoon filter':  # New option selected
			image = Image.open(image_file)
			img_array = np.array(image)
			cartoon_image = apply_cartoon_filter(img_array)  # Apply cartoon filter
			cartoon_pil = Image.fromarray(cartoon_image)  # Convert NumPy array to PIL image
			col1, col2 = st.columns(2)
			with col1:
				st.header("Original Image")
				st.image(load_an_image(image_file), width=250)
			
			with col2:
				st.header("Cartoon Filter")
				st.image(cartoon_pil, width=250)
				buf = BytesIO()
				img = cartoon_pil
				img.save(buf, format="JPEG")
				byte_im = buf.getvalue()
				st.download_button(
                    label="Download image",
                    data=byte_im,
                    file_name="cartoon_image.png",  # Change file name for cartoon image
                    mime="image/png")

if __name__ == '__main__': 
	main() 