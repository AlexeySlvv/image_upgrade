import cv2
from cv2 import dnn_superres

# Read image
finput = 'input'
fext = 'jpg'
image = cv2.imread(f'{finput}.{fext}')

# Read the desired model
model_map = {
    2: 'EDSR_x2.pb',
    # 3: 'EDSR_x3.pb',
    # 4: 'EDSR_x4.pb'
}
for k, v in model_map.items():
    print('scale', k)

    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(v)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", k)

    # Upscale the image
    result = sr.upsample(image)

    # Save the image
    cv2.imwrite(f"{finput}_x{k}.{fext}", result)
