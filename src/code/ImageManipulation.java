package code;

import image.APImage;
import image.Pixel;
import java.util.Random;

public class ImageManipulation {

    /** CHALLENGE 0: Display Image
     *  Write a statement that will display the image in a window
     */
    public static void main(String[] args) {
        String imagePath = "cyberpunk2077.jpg";
        
        // Uncomment one of the following to test different image manipulations:
        
        // 1. Display original image
        // APImage img = new APImage(imagePath);
        // img.draw();
        
        // 2. Grayscale (using convolution)
        // grayScale(imagePath);
        
        // 3. Black and White (using convolution + thresholding)
        // blackAndWhite(imagePath);
        
        // 4. Edge Detection (using convolution kernels)
        // edgeDetection(imagePath, 20);
        
        // 5. Reflect Image
        // reflectImage(imagePath);
        
        // 6. Rotate Image 90° clockwise
        // rotateImage(imagePath);
        
        // 7. Validate FFT convolution matches spatial convolution
        // validateConvolutions(imagePath);
        
        // Default: Show grayscale using convolution
        grayScale(imagePath);
    }

    /** CHALLENGE ONE: Grayscale (USING CONVOLUTION)
     *
     * INPUT: the complete path file name of the image
     * OUTPUT: a grayscale copy of the image
     *
     * Uses a 4D convolution kernel where each output channel (R, G, B) receives
     * equal contributions (1/3) from all input channels, producing the average.
     * 
     * Kernel structure: output[c] = (1/3)*R + (1/3)*G + (1/3)*B for all c
     */
    public static void grayScale(String pathOfFile) {
        APImage img = new APImage(pathOfFile);
        
        // Create grayscale kernel [3][3][1][1]: each output channel = average of all input channels
        double[][][][] grayscaleKernel = new double[3][3][1][1];
        for (int outC = 0; outC < 3; outC++) {
            grayscaleKernel[outC][0][0][0] = 1.0 / 3.0;  // Red contribution
            grayscaleKernel[outC][1][0][0] = 1.0 / 3.0;  // Green contribution
            grayscaleKernel[outC][2][0][0] = 1.0 / 3.0;  // Blue contribution
        }
        
        // Apply convolution
        double[][][] output = convolve2D(img, grayscaleKernel, 1, 0);
        
        // Convert to image and display
        APImage result = toImage(output);
        result.draw();
    }

    /** A helper method that can be used to assist you in each challenge.
     * This method simply calculates the average of the RGB values of a single pixel.
     * @param pixel
     * @return the average RGB value
     */
    private static int getAverageColour(Pixel pixel) {
        return (pixel.getRed() + pixel.getGreen() + pixel.getBlue()) / 3;
    }

    /** CHALLENGE TWO: Black and White (USING CONVOLUTION + THRESHOLDING)
     *
     * INPUT: the complete path file name of the image
     * OUTPUT: a black and white copy of the image
     *
     * Step 1: Use convolve2D() with grayscale kernel to compute average brightness
     * Step 2: Apply thresholding (non-linear operation, done post-convolution)
     *         If average < 128: black (0, 0, 0)
     *         If average >= 128: white (255, 255, 255)
     */
    public static void blackAndWhite(String pathOfFile) {
        APImage img = new APImage(pathOfFile);
        
        // Create grayscale kernel [3][3][1][1]
        double[][][][] grayscaleKernel = new double[3][3][1][1];
        for (int outC = 0; outC < 3; outC++) {
            grayscaleKernel[outC][0][0][0] = 1.0 / 3.0;
            grayscaleKernel[outC][1][0][0] = 1.0 / 3.0;
            grayscaleKernel[outC][2][0][0] = 1.0 / 3.0;
        }
        
        // Step 1: Apply grayscale convolution using convolve2D()
        double[][][] grayscale = convolve2D(img, grayscaleKernel, 1, 0);
        
        int height = grayscale[0].length;
        int width = grayscale[0][0].length;
        
        // Step 2: Apply thresholding (non-linear operation)
        APImage result = new APImage(width, height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // After grayscale convolution, all channels have same value
                double average = grayscale[0][y][x];
                
                // Threshold: < 128 -> black, >= 128 -> white
                int value = (average < 128) ? 0 : 255;
                result.getPixel(x, y).setRed(value);
                result.getPixel(x, y).setGreen(value);
                result.getPixel(x, y).setBlue(value);
            }
        }
        
        result.draw();
    }

    /** CHALLENGE Three: Edge Detection (USING CONVOLUTION)
     *
     * INPUT: the complete path file name of the image
     * OUTPUT: an outline of the image. The amount of information will correspond to the threshold.
     *
     * Uses convolve2D() with two gradient kernels:
     * 1. Horizontal gradient kernel (1x2): computes (current - left) grayscale difference
     * 2. Vertical gradient kernel (2x1): computes (current - below) grayscale difference
     *
     * If either gradient exceeds the threshold, the pixel is marked as an edge (black).
     */
    public static void edgeDetection(String pathToFile, int threshold) {
        APImage img = new APImage(pathToFile);
        
        // Create horizontal edge kernel [3][3][1][2] (current - left)
        // Computes: (1/3)(R+G+B)_current - (1/3)(R+G+B)_left
        double[][][][] horizontalKernel = new double[3][3][1][2];
        for (int outC = 0; outC < 3; outC++) {
            for (int inC = 0; inC < 3; inC++) {
                horizontalKernel[outC][inC][0][0] = -1.0 / 3.0;  // Left pixel (negative)
                horizontalKernel[outC][inC][0][1] = 1.0 / 3.0;   // Current pixel (positive)
            }
        }
        
        // Create vertical edge kernel [3][3][2][1] (current - below)
        // Computes: (1/3)(R+G+B)_current - (1/3)(R+G+B)_below
        double[][][][] verticalKernel = new double[3][3][2][1];
        for (int outC = 0; outC < 3; outC++) {
            for (int inC = 0; inC < 3; inC++) {
                verticalKernel[outC][inC][0][0] = 1.0 / 3.0;   // Current pixel (positive)
                verticalKernel[outC][inC][1][0] = -1.0 / 3.0;  // Below pixel (negative)
            }
        }
        
        // Apply horizontal gradient convolution
        double[][][] hGradient = convolve2D(img, horizontalKernel, 1, 0);
        
        // Apply vertical gradient convolution
        double[][][] vGradient = convolve2D(img, verticalKernel, 1, 0);
        
        // Create result image with dimensions that work for both gradients
        int outWidth = hGradient[0][0].length;   // width - 1
        int outHeight = vGradient[0].length;      // height - 1
        APImage result = new APImage(outWidth, outHeight);
        
        // Combine gradients and apply threshold
        for (int y = 0; y < outHeight; y++) {
            for (int x = 0; x < outWidth; x++) {
                // Get absolute gradients (channel 0, all channels same after grayscale-style kernel)
                double hAbs = Math.abs(hGradient[0][y][x]);
                double vAbs = Math.abs(vGradient[0][y][x]);
                
                // If either gradient exceeds threshold, it's an edge (black), else white
                boolean isEdge = hAbs > threshold || vAbs > threshold;
                int value = isEdge ? 0 : 255;
                
                result.getPixel(x, y).setRed(value);
                result.getPixel(x, y).setGreen(value);
                result.getPixel(x, y).setBlue(value);
            }
        }
        
        result.draw();
    }

    /** CHALLENGE Four: Reflect Image
     *
     * INPUT: the complete path file name of the image
     * OUTPUT: the image reflected about the y-axis
     *
     * Swaps pixels horizontally: pixel at (x, y) moves to (width - 1 - x, y)
     */
    public static void reflectImage(String pathToFile) {
        APImage img = new APImage(pathToFile);
        int width = img.getWidth();
        int height = img.getHeight();
        
        // Only need to iterate through half the width to swap
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width / 2; x++) {
                // Get pixels from both sides
                Pixel left = img.getPixel(x, y);
                Pixel right = img.getPixel(width - 1 - x, y);
                
                // Store left pixel values
                int leftR = left.getRed();
                int leftG = left.getGreen();
                int leftB = left.getBlue();
                
                // Copy right to left
                left.setRed(right.getRed());
                left.setGreen(right.getGreen());
                left.setBlue(right.getBlue());
                
                // Copy stored left values to right
                right.setRed(leftR);
                right.setGreen(leftG);
                right.setBlue(leftB);
            }
        }
        
        img.draw();
    }

    /** CHALLENGE Five: Rotate Image
     *
     * INPUT: the complete path file name of the image
     * OUTPUT: the image rotated 90 degrees CLOCKWISE
     *
     * For 90° clockwise rotation:
     * - New width = old height
     * - New height = old width
     * - Pixel at (x, y) moves to (height - 1 - y, x)
     */
    public static void rotateImage(String pathToFile) {
        APImage img = new APImage(pathToFile);
        int oldWidth = img.getWidth();
        int oldHeight = img.getHeight();
        
        // Create new image with swapped dimensions
        APImage rotated = new APImage(oldHeight, oldWidth);
        
        // Copy pixels with rotation transformation
        for (int y = 0; y < oldHeight; y++) {
            for (int x = 0; x < oldWidth; x++) {
                Pixel original = img.getPixel(x, y);
                
                // 90° clockwise: (x, y) -> (oldHeight - 1 - y, x)
                int newX = oldHeight - 1 - y;
                int newY = x;
                
                Pixel target = rotated.getPixel(newX, newY);
                target.setRed(original.getRed());
                target.setGreen(original.getGreen());
                target.setBlue(original.getBlue());
            }
        }
        
        rotated.draw();
    }

    /** CHALLENGE SIX: 2D Convolution
     *
     * INPUT: an APImage, a 4D kernel array, stride, and padding
     * OUTPUT: raw convolution result as double[3][height][width]
     *
     * Applies a 2D convolution operation across the image using the provided kernel.
     * The kernel is a 4D array with shape [3][3][k][k]:
     *   - First dimension (3): output channels (R, G, B)
     *   - Second dimension (3): input channels (R, G, B)
     *   - Third/Fourth dimensions (k x k): spatial kernel size
     *
     * The convolution slides across the image with the given stride and padding.
     * For each output position, we compute:
     *   output[outC][y][x] = SUM over (inC, ky, kx) of:
     *                        input[inC][y*stride+ky][x*stride+kx] * kernel[outC][inC][ky][kx]
     *
     * No bias is applied.
     *
     * @param img the input APImage
     * @param kernel 4D convolution kernel [outChannels=3][inChannels=3][kernelH][kernelW]
     * @param stride the stride for the convolution (symmetric on height and width)
     * @param padding the padding to apply (symmetric on all sides)
     * @return 3D array [channel][height][width] of raw convolution results
     */
    public static double[][][] convolve2D(APImage img, double[][][][] kernel, int stride, int padding) {
        int width = img.getWidth();
        int height = img.getHeight();
        
        // Extract kernel dimensions
        int kernelHeight = kernel[0][0].length;
        int kernelWidth = kernel[0][0][0].length;
        
        // Calculate output dimensions using standard convolution formula:
        // output_dim = floor((input_dim + 2*padding - kernel_dim) / stride) + 1
        int outHeight = (height + 2 * padding - kernelHeight) / stride + 1;
        int outWidth = (width + 2 * padding - kernelWidth) / stride + 1;
        
        // Create padded input array [channel][height][width]
        // Channel 0 = Red, Channel 1 = Green, Channel 2 = Blue
        double[][][] paddedInput = new double[3][height + 2 * padding][width + 2 * padding];
        
        // Fill padded input with image data (padding regions stay as 0 - zero padding)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Pixel p = img.getPixel(x, y);
                paddedInput[0][y + padding][x + padding] = p.getRed();
                paddedInput[1][y + padding][x + padding] = p.getGreen();
                paddedInput[2][y + padding][x + padding] = p.getBlue();
            }
        }
        
        // Create output array [channel][height][width]
        double[][][] output = new double[3][outHeight][outWidth];
        
        // Perform convolution - 6 nested loops for the full 2D conv operation
        for (int outC = 0; outC < 3; outC++) {              // For each output channel (R, G, B)
            for (int outY = 0; outY < outHeight; outY++) {  // For each output row
                for (int outX = 0; outX < outWidth; outX++) { // For each output column
                    double sum = 0.0;
                    
                    // Compute the convolution at this spatial position
                    for (int inC = 0; inC < 3; inC++) {              // For each input channel
                        for (int ky = 0; ky < kernelHeight; ky++) {  // For each kernel row
                            for (int kx = 0; kx < kernelWidth; kx++) { // For each kernel column
                                // Map output position to input position using stride
                                int inY = outY * stride + ky;
                                int inX = outX * stride + kx;
                                // Accumulate: input value * kernel weight
                                sum += paddedInput[inC][inY][inX] * kernel[outC][inC][ky][kx];
                            }
                        }
                    }
                    
                    output[outC][outY][outX] = sum;
                }
            }
        }
        
        return output;
    }

    /**
     * Converts raw convolution output to an APImage.
     * Clamps values to valid [0, 255] pixel range.
     *
     * @param convOutput the raw convolution output [3][height][width]
     * @return an APImage with the convolution results
     */
    private static APImage toImage(double[][][] convOutput) {
        int height = convOutput[0].length;
        int width = convOutput[0][0].length;
        APImage result = new APImage(width, height);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = clamp((int) Math.round(convOutput[0][y][x]), 0, 255);
                int g = clamp((int) Math.round(convOutput[1][y][x]), 0, 255);
                int b = clamp((int) Math.round(convOutput[2][y][x]), 0, 255);
                result.setPixel(x, y, new Pixel(r, g, b));
            }
        }
        
        return result;
    }

    /**
     * Helper method to clamp a value to a specified range.
     */
    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    // ==================== FFT-BASED FAST CONVOLUTION ====================

    /**
     * Fast 2D Convolution using FFT (Fast Fourier Transform).
     * 
     * Uses the convolution theorem: convolution in spatial domain equals
     * element-wise multiplication in frequency domain.
     * 
     * Time complexity: O(N^2 log N) instead of O(N^2 * K^2) for spatial convolution
     * where N is image size and K is kernel size.
     * 
     * Note: FFT convolution only supports stride=1. For other strides, 
     * subsample the output after convolution.
     *
     * @param img the input APImage
     * @param kernel 4D convolution kernel [outChannels=3][inChannels=3][kernelH][kernelW]
     * @param stride the stride (1 for full FFT speed, >1 will subsample output)
     * @param padding the padding to apply (symmetric on all sides)
     * @return 3D array [channel][height][width] of raw convolution results
     */
    public static double[][][] fastConvolve2D(APImage img, double[][][][] kernel, int stride, int padding) {
        int width = img.getWidth();
        int height = img.getHeight();
        int kernelHeight = kernel[0][0].length;
        int kernelWidth = kernel[0][0][0].length;
        
        // Padded input dimensions
        int paddedHeight = height + 2 * padding;
        int paddedWidth = width + 2 * padding;
        
        // FFT size: must be power of 2, and large enough for linear (non-circular) convolution
        // Size = input + kernel - 1, then round up to next power of 2
        int fftHeight = nextPowerOf2(paddedHeight + kernelHeight - 1);
        int fftWidth = nextPowerOf2(paddedWidth + kernelWidth - 1);
        
        // Extract padded input channels
        double[][][] paddedInput = new double[3][paddedHeight][paddedWidth];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Pixel p = img.getPixel(x, y);
                paddedInput[0][y + padding][x + padding] = p.getRed();
                paddedInput[1][y + padding][x + padding] = p.getGreen();
                paddedInput[2][y + padding][x + padding] = p.getBlue();
            }
        }
        
        // Pre-compute FFT of all input channels
        double[][][][] inputFFT = new double[3][2][fftHeight][fftWidth]; // [channel][real/imag][h][w]
        for (int inC = 0; inC < 3; inC++) {
            double[][] realPart = new double[fftHeight][fftWidth];
            double[][] imagPart = new double[fftHeight][fftWidth];
            
            // Copy padded input into FFT buffer (zero-padded to fftSize)
            for (int y = 0; y < paddedHeight; y++) {
                for (int x = 0; x < paddedWidth; x++) {
                    realPart[y][x] = paddedInput[inC][y][x];
                }
            }
            
            // Compute 2D FFT
            fft2D(realPart, imagPart, false);
            inputFFT[inC][0] = realPart;
            inputFFT[inC][1] = imagPart;
        }
        
        // Output dimensions (before stride)
        int outHeightFull = paddedHeight - kernelHeight + 1;
        int outWidthFull = paddedWidth - kernelWidth + 1;
        
        // Output dimensions (after stride)
        int outHeight = (outHeightFull - 1) / stride + 1;
        int outWidth = (outWidthFull - 1) / stride + 1;
        
        double[][][] output = new double[3][outHeight][outWidth];
        
        // For each output channel
        for (int outC = 0; outC < 3; outC++) {
            // Accumulator in frequency domain for this output channel
            double[][] accumReal = new double[fftHeight][fftWidth];
            double[][] accumImag = new double[fftHeight][fftWidth];
            
            // Sum over input channels
            for (int inC = 0; inC < 3; inC++) {
                // Prepare kernel for FFT (flip for convolution, not correlation)
                double[][] kernelReal = new double[fftHeight][fftWidth];
                double[][] kernelImag = new double[fftHeight][fftWidth];
                
                // Copy flipped kernel (180° rotation for convolution)
                for (int ky = 0; ky < kernelHeight; ky++) {
                    for (int kx = 0; kx < kernelWidth; kx++) {
                        kernelReal[ky][kx] = kernel[outC][inC][kernelHeight - 1 - ky][kernelWidth - 1 - kx];
                    }
                }
                
                // Compute 2D FFT of kernel
                fft2D(kernelReal, kernelImag, false);
                
                // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                for (int y = 0; y < fftHeight; y++) {
                    for (int x = 0; x < fftWidth; x++) {
                        double a = inputFFT[inC][0][y][x];
                        double b = inputFFT[inC][1][y][x];
                        double c = kernelReal[y][x];
                        double d = kernelImag[y][x];
                        
                        accumReal[y][x] += a * c - b * d;
                        accumImag[y][x] += a * d + b * c;
                    }
                }
            }
            
            // Inverse FFT
            fft2D(accumReal, accumImag, true);
            
            // Extract valid region with stride
            for (int y = 0; y < outHeight; y++) {
                for (int x = 0; x < outWidth; x++) {
                    int srcY = y * stride + kernelHeight - 1;
                    int srcX = x * stride + kernelWidth - 1;
                    output[outC][y][x] = accumReal[srcY][srcX];
                }
            }
        }
        
        return output;
    }

    /**
     * Returns the next power of 2 >= n.
     */
    private static int nextPowerOf2(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }

    /**
     * In-place 2D FFT using row-column decomposition.
     * 
     * @param real the real part of the 2D signal (modified in place)
     * @param imag the imaginary part of the 2D signal (modified in place)
     * @param inverse true for inverse FFT, false for forward FFT
     */
    private static void fft2D(double[][] real, double[][] imag, boolean inverse) {
        int height = real.length;
        int width = real[0].length;
        
        // FFT on each row
        double[] rowReal = new double[width];
        double[] rowImag = new double[width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                rowReal[x] = real[y][x];
                rowImag[x] = imag[y][x];
            }
            fft1D(rowReal, rowImag, inverse);
            for (int x = 0; x < width; x++) {
                real[y][x] = rowReal[x];
                imag[y][x] = rowImag[x];
            }
        }
        
        // FFT on each column
        double[] colReal = new double[height];
        double[] colImag = new double[height];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                colReal[y] = real[y][x];
                colImag[y] = imag[y][x];
            }
            fft1D(colReal, colImag, inverse);
            for (int y = 0; y < height; y++) {
                real[y][x] = colReal[y];
                imag[y][x] = colImag[y];
            }
        }
    }

    /**
     * In-place 1D FFT using Cooley-Tukey radix-2 algorithm.
     * 
     * @param real the real part of the signal (modified in place)
     * @param imag the imaginary part of the signal (modified in place)
     * @param inverse true for inverse FFT, false for forward FFT
     */
    private static void fft1D(double[] real, double[] imag, boolean inverse) {
        int n = real.length;
        if (n == 0) return;
        
        // Bit-reversal permutation
        int bits = Integer.numberOfTrailingZeros(n);
        for (int i = 0; i < n; i++) {
            int j = Integer.reverse(i) >>> (32 - bits);
            if (i < j) {
                double tempR = real[i];
                double tempI = imag[i];
                real[i] = real[j];
                imag[i] = imag[j];
                real[j] = tempR;
                imag[j] = tempI;
            }
        }
        
        // Cooley-Tukey iterative FFT
        for (int size = 2; size <= n; size *= 2) {
            int halfSize = size / 2;
            double angle = (inverse ? 2.0 : -2.0) * Math.PI / size;
            double wReal = Math.cos(angle);
            double wImag = Math.sin(angle);
            
            for (int start = 0; start < n; start += size) {
                double curReal = 1.0;
                double curImag = 0.0;
                
                for (int k = 0; k < halfSize; k++) {
                    int even = start + k;
                    int odd = start + k + halfSize;
                    
                    // Butterfly operation
                    double tReal = curReal * real[odd] - curImag * imag[odd];
                    double tImag = curReal * imag[odd] + curImag * real[odd];
                    
                    real[odd] = real[even] - tReal;
                    imag[odd] = imag[even] - tImag;
                    real[even] = real[even] + tReal;
                    imag[even] = imag[even] + tImag;
                    
                    // Update twiddle factor
                    double newReal = curReal * wReal - curImag * wImag;
                    double newImag = curReal * wImag + curImag * wReal;
                    curReal = newReal;
                    curImag = newImag;
                }
            }
        }
        
        // Scale for inverse FFT
        if (inverse) {
            for (int i = 0; i < n; i++) {
                real[i] /= n;
                imag[i] /= n;
            }
        }
    }

    // ==================== VALIDATION ====================

    /**
     * Validates that convolve2D and fastConvolve2D produce the same results
     * within numerical tolerance for random normally-distributed kernels.
     * 
     * Tests multiple kernel sizes and reports max error for each.
     * 
     * @param imagePath path to test image
     */
    public static void validateConvolutions(String imagePath) {
        APImage img = new APImage(imagePath);
        Random rng = new Random(42);  // Fixed seed for reproducibility
        
        // Kernel sizes to test
        int[] kernelSizes = {1, 3, 5, 7, 9, 11};
        
        // Tolerance for floating point comparison
        double tolerance = 1e-6;
        
        System.out.println("=== Convolution Validation: convolve2D vs fastConvolve2D ===");
        System.out.println("Image size: " + img.getWidth() + " x " + img.getHeight());
        System.out.println("Tolerance: " + tolerance);
        System.out.println();
        
        boolean allPassed = true;
        
        for (int kernelSize : kernelSizes) {
            // Test with multiple random kernels per size
            int numTrials = 3;
            double maxErrorForSize = 0.0;
            
            for (int trial = 0; trial < numTrials; trial++) {
                // Generate random kernel from normal distribution
                double[][][][] kernel = generateRandomKernel(kernelSize, kernelSize, rng);
                
                // Test with padding = kernelSize/2 (common "same" padding)
                int padding = kernelSize / 2;
                
                // Run both convolutions
                double[][][] spatialResult = convolve2D(img, kernel, 1, padding);
                double[][][] fftResult = fastConvolve2D(img, kernel, 1, padding);
                
                // Compare results
                double maxError = compareOutputs(spatialResult, fftResult);
                maxErrorForSize = Math.max(maxErrorForSize, maxError);
            }
            
            // Report results
            boolean passed = maxErrorForSize < tolerance;
            String status = passed ? "PASS" : "FAIL";
            System.out.printf("Kernel %2dx%-2d : Max Error = %.2e  [%s]%n", 
                              kernelSize, kernelSize, maxErrorForSize, status);
            
            if (!passed) {
                allPassed = false;
            }
        }
        
        System.out.println();
        
        // Additional test: non-square kernels
        System.out.println("--- Testing non-square kernels ---");
        int[][] nonSquareSizes = {{1, 3}, {3, 1}, {3, 5}, {5, 3}, {1, 7}, {7, 1}};
        
        for (int[] size : nonSquareSizes) {
            int kh = size[0];
            int kw = size[1];
            
            double[][][][] kernel = generateRandomKernel(kh, kw, rng);
            int padH = kh / 2;
            int padW = kw / 2;
            int padding = Math.max(padH, padW);
            
            double[][][] spatialResult = convolve2D(img, kernel, 1, padding);
            double[][][] fftResult = fastConvolve2D(img, kernel, 1, padding);
            
            double maxError = compareOutputs(spatialResult, fftResult);
            boolean passed = maxError < tolerance;
            String status = passed ? "PASS" : "FAIL";
            System.out.printf("Kernel %2dx%-2d : Max Error = %.2e  [%s]%n", 
                              kh, kw, maxError, status);
            
            if (!passed) {
                allPassed = false;
            }
        }
        
        System.out.println();
        
        // Performance comparison
        System.out.println("--- Performance Comparison (single run) ---");
        double[][][][] testKernel = generateRandomKernel(7, 7, rng);
        
        long startSpatial = System.nanoTime();
        convolve2D(img, testKernel, 1, 3);
        long endSpatial = System.nanoTime();
        
        long startFFT = System.nanoTime();
        fastConvolve2D(img, testKernel, 1, 3);
        long endFFT = System.nanoTime();
        
        double spatialMs = (endSpatial - startSpatial) / 1_000_000.0;
        double fftMs = (endFFT - startFFT) / 1_000_000.0;
        
        System.out.printf("Spatial convolution (7x7 kernel): %.2f ms%n", spatialMs);
        System.out.printf("FFT convolution (7x7 kernel):     %.2f ms%n", fftMs);
        System.out.printf("Speedup: %.2fx%n", spatialMs / fftMs);
        
        System.out.println();
        System.out.println("=== Overall: " + (allPassed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") + " ===");
    }

    /**
     * Generates a random 4D kernel [3][3][kh][kw] with values from standard normal distribution.
     * 
     * @param kh kernel height
     * @param kw kernel width
     * @param rng random number generator
     * @return random kernel
     */
    private static double[][][][] generateRandomKernel(int kh, int kw, Random rng) {
        double[][][][] kernel = new double[3][3][kh][kw];
        
        for (int outC = 0; outC < 3; outC++) {
            for (int inC = 0; inC < 3; inC++) {
                for (int y = 0; y < kh; y++) {
                    for (int x = 0; x < kw; x++) {
                        // Standard normal distribution (mean=0, std=1)
                        kernel[outC][inC][y][x] = rng.nextGaussian();
                    }
                }
            }
        }
        
        return kernel;
    }

    /**
     * Compares two convolution outputs and returns the maximum absolute error.
     * 
     * @param a first output [3][h][w]
     * @param b second output [3][h][w]
     * @return maximum absolute difference
     */
    private static double compareOutputs(double[][][] a, double[][][] b) {
        double maxError = 0.0;
        
        // Check dimensions match
        if (a.length != b.length || a[0].length != b[0].length || a[0][0].length != b[0][0].length) {
            System.err.println("WARNING: Output dimensions don't match!");
            System.err.println("  A: [" + a.length + "][" + a[0].length + "][" + a[0][0].length + "]");
            System.err.println("  B: [" + b.length + "][" + b[0].length + "][" + b[0][0].length + "]");
            return Double.MAX_VALUE;
        }
        
        int channels = a.length;
        int height = a[0].length;
        int width = a[0][0].length;
        
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double error = Math.abs(a[c][y][x] - b[c][y][x]);
                    maxError = Math.max(maxError, error);
                }
            }
        }
        
        return maxError;
    }

}
