using GemmForge.Common;
using GemmForge.Gpu;
using NUnit.Framework;

namespace GemmForge
{
    [TestFixture]
    public class Tests
    {
        [Test]
        public void TestDeclareVariable()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            
            var matrixA = new Variable(new SinglePrecisionFloat(), "A");
            var init = new Assignment(new Literal("5"));
            
            var code = builder.DeclareVariable(matrixA, init).Build();
            Assert.AreEqual("float A = 5;\n", code.ToString());
        }
        
        [Test]
        public void TestDeclarePointer()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            
            var matrixA = new Variable(new SinglePrecisionFloat(), "A");
            var init = new Assignment(new Literal("5"));
            
            var code = builder.DeclarePointer(matrixA, init).Build();
            Assert.AreEqual("float *A = 5;\n", code.ToString());
        }
        
        [Test]
        public void TestDeclareSyclDevicePointer()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            
            var matrixA = new Variable(new SinglePrecisionFloat(), "A");
            var init = new Assignment(new MallocShared(new SinglePrecisionFloat(), new Literal("5")));
            
            var code = builder.DeclarePointer(matrixA, init).Build();
            Assert.AreEqual("float *A = malloc_device<float>(5);\n", code.ToString());
        }
        
        [Test]
        public void TestDeclareCArray()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            
            var matrixA = new Variable(new SinglePrecisionFloat(), "A");
            var five = new Literal("5");
            var code = builder.DeclareArray(matrixA, five).Build();
            Assert.AreEqual("float A[5];\n", code.ToString());
        }
        
        [Test]
        public void TestDeclareCudaSharedMemory()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            
            var matrixA = new Variable(new SharedVariableType(new SinglePrecisionFloat()), "A");
            var code = builder.DeclareArray(matrixA, new Literal("5")).Build();
            Assert.AreEqual("__shared__ float A[5];\n", code.ToString());
        }
        
        [Test]
        public void TestDeclareCudaSharedMemoryWithAddition()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            
            var matrixA = new Variable(new SharedVariableType(new SinglePrecisionFloat()), "A");
            var code = builder.DeclareArray(matrixA, new Addition(new Literal("5"), new Literal("10"))).Build();
            Assert.AreEqual("__shared__ float A[5 + 10];\n", code.ToString());
        }
    }
}