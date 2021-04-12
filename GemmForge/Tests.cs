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
        public void TestDeclareCArray()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            
            var matrixA = new Variable(new SinglePrecisionFloat(), "A");
            var five = new Literal("5");
            var code = builder.DeclareArray(matrixA, five).Build();
            Assert.AreEqual("float A[5];\n", code.ToString());
        }
        
        [Test]
        public void TestDeclareCudaSharedLocalMemory()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            
            var matrixA = new Variable(new SinglePrecisionFloat(), "A");
            var code = builder.MallocGpuSharedMemory(new Malloc(matrixA, new Literal("5"), MallocHints.MallocLocal)).Build();
            Assert.AreEqual("__shared__ float A[5];\n", code.ToString());
        }
        
        [Test]
        public void TestDeclareCudaSharedLocalMemoryWithAddition()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            
            var matrixA = new Variable(new SinglePrecisionFloat(), "A");
            var malloc = new Malloc(matrixA, new Addition(new Literal("5"), new Literal("10")), MallocHints.MallocLocal);
            var code = builder.MallocGpuSharedMemory(malloc).Build();
            Assert.AreEqual("__shared__ float A[5 + 10];\n", code.ToString());
        }
        
        [Test]
        public void TestDeclareCudaSharedGlobalMemory()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            
            var matrixA = new Variable(new SinglePrecisionFloat(), "A");
            var malloc = new Malloc(matrixA, new Literal("5"));
            var code = builder.MallocGpuSharedMemory(malloc).Build();
            Assert.AreEqual("float *A;\ncudaMallocManaged(&A, 5 * sizeof(float), cudaMemAttachGlobal);\n", code.ToString());
        }
        
        [Test]
        public void TestDeclareSyclSharedLocalMemory()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            
            var matrixA = new Variable(new SinglePrecisionFloat(), "A");
            var code = builder.MallocGpuSharedMemory(new Malloc(matrixA, new Literal("5"), MallocHints.MallocLocal)).Build();
            Assert.AreEqual("float *A = malloc_shared<float>(5);\n", code.ToString());
        } 

        
        [Test]
        public void TestDeclareSyclSharedGlobalMemory()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            
            var matrixA = new Variable(new SinglePrecisionFloat(), "A");
            var code = builder.MallocGpuSharedMemory(new Malloc(matrixA, new Literal("5"))).Build();
            Assert.AreEqual("float *A = malloc_shared<float>(5);\n", code.ToString());
        }
        
        [Test]
        public void TestDeclareWithoutInit()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            
            var matrixA = new Variable(new SinglePrecisionFloat(), "A");
            var code = builder.DeclarePointer(matrixA).Build();
            
            Assert.AreEqual("float *A;\n", code.ToString());
        }
        

        
    }
}