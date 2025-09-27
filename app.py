from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("food_vision.h5")

class_names=['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

# Image preprocessing (same as training pipeline)
def load_and_prep_image_bytes(image_bytes, img_shape=224):
    img = tf.io.decode_image(image_bytes, channels=3)   # decode bytes
    img = tf.image.resize(img, [img_shape, img_shape])
    
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Read file as bytes (no saving)
            img_bytes = file.read()
            img_tensor = load_and_prep_image_bytes(img_bytes)

            # Predict
            pred = model.predict(tf.expand_dims(img_tensor, axis=0))
            pred_class = class_names[np.argmax(pred)]

            # Convert image to base64 for display
            img_data = base64.b64encode(img_bytes).decode("utf-8")
            img_data = f"data:image/jpeg;base64,{img_data}"

            return render_template("index.html", prediction=pred_class, img_data=img_data)

    return render_template("index.html", prediction=None)
    
if __name__ == "__main__":
    app.run(debug=True)


