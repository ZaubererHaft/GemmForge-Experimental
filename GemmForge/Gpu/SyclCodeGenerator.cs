namespace GemmForge.Gpu
{
    public class SyclCodeGenerator : IGPUCodeGenerator
    {
        private readonly Code _code;

        public SyclCodeGenerator(Code code)
        {
            _code = code;
        }

    }
}