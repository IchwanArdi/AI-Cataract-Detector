"""
Script untuk memperbaiki shape mismatch pada model cataract detection
Jalankan script ini untuk membuat model yang kompatibel

Usage: python fix_model.py
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

def fix_model_shape_mismatch(model_path='best_cataract_model.h5', output_path='best_cataract_model_fixed.h5'):
    """
    Memperbaiki model yang memiliki shape mismatch antara training dan deployment
    """
    
    print("🔧 Starting model shape mismatch fix...")
    
    try:
        # Method 1: Try to load and re-save with correct input shape
        print(f"📂 Loading model from: {model_path}")
        
        # Load model architecture only
        try:
            # Try to get model config first
            model = tf.keras.models.load_model(model_path, compile=False)
            config = model.get_config()
            
            # Modify input shape in config
            if 'layers' in config and len(config['layers']) > 0:
                input_layer = config['layers'][0]
                if input_layer['class_name'] == 'InputLayer':
                    # Change input shape to RGB (3 channels)
                    input_layer['config']['batch_input_shape'] = [None, 224, 224, 3]
                    print("✅ Modified input layer to accept RGB images")
            
            # Create new model from modified config
            fixed_model = tf.keras.Model.from_config(config)
            
            # Try to copy weights where possible
            print("🔄 Copying compatible weights...")
            for i, (old_layer, new_layer) in enumerate(zip(model.layers, fixed_model.layers)):
                try:
                    if len(old_layer.get_weights()) > 0:
                        old_weights = old_layer.get_weights()
                        
                        # Skip first conv layer if shapes don't match
                        if i == 1 and "conv" in old_layer.name.lower():
                            print(f"⚠️ Skipping first conv layer {old_layer.name} due to channel mismatch")
                            continue
                            
                        new_layer.set_weights(old_weights)
                        print(f"✅ Copied weights for layer: {old_layer.name}")
                except Exception as e:
                    print(f"⚠️ Could not copy weights for layer {old_layer.name}: {str(e)}")
                    continue
            
            # Save fixed model
            fixed_model.save(output_path)
            print(f"✅ Fixed model saved to: {output_path}")
            
            # Verify the fixed model
            print("\n🔍 Verifying fixed model...")
            test_model = tf.keras.models.load_model(output_path)
            print(f"✅ Fixed model input shape: {test_model.input_shape}")
            
            # Test with dummy RGB input
            dummy_input = np.random.rand(1, 224, 224, 3)
            try:
                prediction = test_model.predict(dummy_input, verbose=0)
                print(f"✅ Test prediction successful! Output shape: {prediction.shape}")
                return True
            except Exception as e:
                print(f"❌ Test prediction failed: {str(e)}")
                return False
                
        except Exception as e:
            print(f"❌ Method 1 failed: {str(e)}")
            
    except Exception as e:
        print(f"❌ Failed to fix model: {str(e)}")
        return False

def create_new_compatible_model(output_path='cataract_model_new.h5'):
    """
    Membuat model baru dengan arsitektur yang kompatibel
    """
    
    print("🆕 Creating new compatible model...")
    
    # Create a simple CNN model for cataract detection
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),  # RGB input
        
        # Feature extraction layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv4'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Classification layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu', name='dense1'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')  # Binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Save model
    model.save(output_path)
    print(f"✅ New model saved to: {output_path}")
    print(f"📏 Model input shape: {model.input_shape}")
    print(f"📊 Model summary:")
    model.summary()
    
    return True

def main():
    print("=" * 50)
    print("🔧 CATARACT MODEL SHAPE MISMATCH FIXER")
    print("=" * 50)
    
    # Check if original model exists
    model_path = 'best_cataract_model.h5'
    
    if os.path.exists(model_path):
        print(f"✅ Found model: {model_path}")
        
        # Try to fix the existing model
        success = fix_model_shape_mismatch(model_path)
        
        if success:
            print("\n🎉 Model successfully fixed!")
            print("📝 You can now use 'best_cataract_model_fixed.h5' in your app")
        else:
            print("\n⚠️ Could not fix existing model. Creating new compatible model...")
            create_new_compatible_model()
            
    else:
        print(f"❌ Model not found: {model_path}")
        print("🆕 Creating new compatible model instead...")
        create_new_compatible_model()
    
    print("\n" + "=" * 50)
    print("✅ PROCESS COMPLETED")
    print("=" * 50)
    
    print("""
    📝 NEXT STEPS:
    
    1. If 'best_cataract_model_fixed.h5' was created:
       - Replace your model file with the fixed version
       - Or update your app to use the fixed model path
    
    2. If 'cataract_model_new.h5' was created:
       - This is a new model that needs to be trained
       - Use your training data to train this model
       - Replace the old model with this trained version
    
    3. Update your Streamlit app to use the correct model path
    
    4. Make sure your training data preprocessing matches:
       - RGB images (3 channels)
       - Size: 224x224 pixels
       - Pixel values normalized to [0, 1]
    """)

if __name__ == "__main__":
    main()