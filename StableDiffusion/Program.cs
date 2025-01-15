using StableDiffusion.ML.OnnxRuntime;

namespace StableDiffusion
{
    public class Program
    {
        static void Main(string[] args)
        {
            const string MODELPATH = @"C:\tmp\models\stable-diffusion-v1-4";
            //test how long this takes to execute
            var watch = System.Diagnostics.Stopwatch.StartNew();

            //Default args
            var prompt = "woman knitting a cocoon";
            Console.WriteLine(prompt);

            var config = new StableDiffusionConfig
            {
                // Number of denoising steps
                NumInferenceSteps = 15,
                // Scale for classifier-free guidance
                GuidanceScale = 7.5,
                // Set your preferred Execution Provider. Currently (GPU, DirectML, CPU) are supported in this project.
                // ONNX Runtime supports many more than this. Learn more here: https://onnxruntime.ai/docs/execution-providers/
                // The config is defaulted to CUDA. You can override it here if needed.
                // To use DirectML EP intall the Microsoft.ML.OnnxRuntime.DirectML and uninstall Microsoft.ML.OnnxRuntime.GPU
                ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.Cuda,
                // Set GPU Device ID.
                DeviceId = 0,
                // Update paths to your models
                // C:\tmp\models\stable-diffusion-v1-4\text_encoder
                TextEncoderOnnxPath = Path.Combine(MODELPATH, @"text_encoder\model.onnx"),
                UnetOnnxPath = Path.Combine(MODELPATH, @"unet\model.onnx"),
                VaeDecoderOnnxPath = Path.Combine(MODELPATH, @"vae_decoder\model.onnx"),
                SafetyModelPath = Path.Combine(MODELPATH, @"safety_checker\model.onnx")
            };

            // Inference Stable Diff
            var image = UNet.Inference(prompt, config);

            // If image failed or was unsafe it will return null.
            if (image == null)
            {
                Console.WriteLine("Unable to create image, please try again.");
            }
            // Stop the timer
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Time taken: " + elapsedMs + "ms");

        }

    }
}