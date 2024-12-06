package com.example.firstml;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private Interpreter tflite;
    private TextView resultTextView;
    private ImageView imageView;
    private ActivityResultLauncher<Intent> activityResultLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultTextView = findViewById(R.id.resultTextView);
        imageView = findViewById(R.id.imageView);

        // Load the TensorFlow Lite model
        try {
            loadModel();
            resultTextView.setText("Model loaded successfully.");
        } catch (IOException e) {
            e.printStackTrace();
            resultTextView.setText("Failed to load model: " + e.getMessage());
        }

        // Initialize the ActivityResultLauncher
        activityResultLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Uri imageUri = result.getData().getData();
                        if (imageUri != null) {
                            try {
                                // Load the image from the gallery
                                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                                if (bitmap == null) {
                                    resultTextView.setText("Bitmap is null");
                                    return;
                                }
                                imageView.setImageBitmap(bitmap);
                                // Preprocess the image and make a prediction
                                int recognizedDigit = predictDigit(bitmap);
                                resultTextView.setText("Recognized: " + recognizedDigit);
                            } catch (IOException e) {
                                e.printStackTrace();
                                resultTextView.setText("Failed to load image: " + e.getMessage());
                            } catch (Exception e) {
                                e.printStackTrace();
                                resultTextView.setText("An unexpected error occurred: " + e.getMessage());
                            }
                        } else {
                            resultTextView.setText("Image URI is null");
                        }
                    }
                }
        );

        // Request permissions and open the gallery
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        } else {
            openGallery();
        }
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        activityResultLauncher.launch(intent);
    }

    private void loadModel() throws IOException {
        FileInputStream inputStream = getAssets().openFd("digit_operator_recognition_model.tflite").createInputStream();
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = getAssets().openFd("digit_operator_recognition_model.tflite").getStartOffset();
        long declaredLength = getAssets().openFd("digit_operator_recognition_model.tflite").getDeclaredLength();
        tflite = new Interpreter(fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength));
    }

    private Bitmap preprocessImage(Bitmap bitmap) {
        return Bitmap.createScaledBitmap(bitmap, 28, 28, true); // Rescale to 28x28
    }

    private float[][][][] convertBitmapToInputArray(Bitmap bitmap) {
        float[][][][] input = new float[1][28][28][1]; // 1 image, 28x28 pixels, 1 channel
        int[] pixels = new int[28 * 28];
        bitmap.getPixels(pixels, 0, 28, 0, 0, 28, 28);

        for (int i = 0; i < pixels.length; i++) {
            int pixelValue = pixels[i];
            int red = (pixelValue >> 16) & 0xFF;
            int green = (pixelValue >> 8) & 0xFF;
            int blue = pixelValue & 0xFF;
            float gray = (red + green + blue) / 3.0f / 255.0f; // Normalize to [0, 1]
            input[0][i / 28][i % 28][0] = gray; // Assign grayscale value
        }
        return input;
    }

    private int predictDigit(Bitmap bitmap) {
        float[][][][] input = convertBitmapToInputArray(preprocessImage(bitmap)); // Call updated method
        float[][] output = new float[1][10]; // Assuming 10 classes (0-9)
        tflite.run(input, output);
        return argMax(output[0]);
    }

    private int argMax(float[] probabilities) {
        int maxIndex = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openGallery();
            } else {
                resultTextView.setText("Permission denied to read storage.");
            }
        }
    }
}
