using GemmForge.Common;
using GemmForge.Gpu;
using NUnit.Framework;

namespace GemmForge
{
    [TestFixture]
    public class Tests
    {
        [Test]
        public void TestBuildVariable()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            
            var matrixA = new Variable(VariableType.SinglePrecisionFloat, "A");
            var init = new Assignment(new Literal("5"));
            
            var code = builder.Declare(matrixA, init).Build();
            Assert.AreEqual("float A = 5;\n", code.ToString());
        }
        
        [Test]
        public void TestBuildPointerVariable()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            
            var matrixA = new Pointer(VariableType.SinglePrecisionFloat, "A");
            var init = new Assignment(new Literal("5"));
            
            var code = builder.Declare(matrixA, init).Build();
            Assert.AreEqual("float *A = 5;\n", code.ToString());
        }
        
        [Test]
        public void TestBuildDevicePointerVariable()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            
            var matrixA = new Pointer(VariableType.SinglePrecisionFloat, "A");
            var init = new Assignment(new MallocShared(VariableType.DoublePrecisionFloat, 5));
            
            var code = builder.Declare(matrixA, init).Build();
            Assert.AreEqual("float *A = malloc_device<float>(5);\n", code.ToString());
        }
    }
}