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
        
        [Test]
        public void TestSyclDeclareGpuKernelRange()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();

            var varA = new Variable(new DoublePrecisionFloat(), "A");
            var block = new Range("block", new Literal("10"), new Literal("1"), new Literal("1"));
            var grid = new Range("grid", new Addition(varA, new Literal("3")), new Literal("1"), new Literal("1"));
            
            var code = builder.DeclareGpuKernelRange(block, grid).Build();
            var s1 = "range<3> block {10, 1, 1};\n" +
                          "range<3> grid {A + 3, 1, 1};\n";
            
            Assert.AreEqual(s1, code.ToString());
        }
        
        [Test]
        public void TestCudaDeclareGpuKernelRange()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();

            var varA = new Variable(new DoublePrecisionFloat(), "A");
            var block = new Range("block", new Literal("10"), new Literal("1"), new Literal("1"));
            var grid = new Range("grid", new Addition(varA, new Literal("3")), new Literal("1"), new Literal("1"));
            
            var code = builder.DeclareGpuKernelRange(block, grid).Build();
            var s1 = "dim3 block (10, 1, 1);\n" +
                          "dim3 grid (A + 3, 1, 1);\n";
            
            Assert.AreEqual(s1, code.ToString());
        }
        
        [Test]
        public void TestSyclInitStreamByPointer()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            var ptr = new Variable(new VoidType(), "stream");
            
            var q = new Stream("q");
            var code = builder.InitStreamByPointer(q, ptr).Build();
            
            Assert.AreEqual("queue q = static_cast<queue>(stream);\n", code.ToString());
        }
        
        [Test]
        public void TestCudaInitStreamByPointer()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            var ptr = new Variable(new VoidType(), "stream");
            
            var q = new Stream("q");
            var code = builder.InitStreamByPointer(q, ptr).Build();
            
            Assert.AreEqual("cudaStream_t q = static_cast<cudaStream_t>(stream);\n", code.ToString());
        }
        
        [Test]
        public void TestDefineSimpleFunction()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            var methodBuilder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();

            var func = new Function("func", new VoidType(), new FunctionArguments(), methodBuilder);
            var code = builder.DefineFunction(func).Build();

            Assert.AreEqual("void func(){\n}\n", code.ToString());
        }  
        
        [Test]
        public void TestDefineComplexFunction()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            var methodBuilder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();

            var varA = new Variable(new SinglePrecisionFloat(), "A");
            var varB = new Variable(new SinglePrecisionFloat(), "B");

            var func = new Function("func", new SinglePrecisionFloat(), new FunctionArguments(varA, varB), methodBuilder);
            methodBuilder.Return(new Addition(varA, varB));
            
            var code = builder.DefineFunction(func).Build();

            Assert.AreEqual("float func(float A, float B){\nreturn A + B;\n}\n", code.ToString());
        }
        
        [Test]
        public void TestDeclareSyclKernelFunction()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            var methodBuilder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            
            var block = new Range("block", new Literal("10"), new Literal("1"), new Literal("1"));
            var grid = new Range("grid",  new Literal("3"), new Literal("1"), new Literal("1"));
            var stream = new Stream("stream");

            var func = new Function("func", new VoidType(), new FunctionArguments(), methodBuilder);
            var kernelFunc = new KernelFunction(func, block, grid, stream);
            
            var code = builder.DefineGpuKernel(kernelFunc).Build();

            Assert.AreEqual("void func(range<3> block, range<3> grid, queue *stream){\n" +
                                    "stream->submit([&](handler &cgh){" +
                                    "cgh.parallel_for(nd_range<3>{block, grid}, [=](nd_item<3> item){\n" +
                                    "});\n});}\n", code.ToString());
        }  
        
        [Test]
        public void TestDeclareCudaKernelFunction()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            var methodBuilder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            
            var block = new Range("block", new Literal("10"), new Literal("1"), new Literal("1"));
            var grid = new Range("grid",  new Literal("3"), new Literal("1"), new Literal("1"));
            var stream = new Stream("stream");

            var func = new Function("func", new VoidType(), new FunctionArguments(), methodBuilder);
            var kernelFunc = new KernelFunction(func, block, grid, stream);
            
            var code = builder.DefineGpuKernel(kernelFunc).Build();

            Assert.AreEqual("__global__ __launch_bounds__(64) void func(){\n}\n", code.ToString());
        }
        
        [Test]
        public void TestLaunchSyclKernel()
        {
            var builder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            var methodBuilder = new CodeBuilderFactory().CreateCppSyclCodeBuilder();
            
            var block = new Range("block", new Literal("10"), new Literal("1"), new Literal("1"));
            var grid = new Range("grid",  new Literal("3"), new Literal("1"), new Literal("1"));
            var stream = new Stream("stream");

            var func = new Function("func", new VoidType(), new FunctionArguments(), methodBuilder);
            var kernelFunc = new KernelFunction(func, block, grid, stream);

            var code = builder.LaunchGpuKernel(kernelFunc).Build();
            Assert.AreEqual("func(block, grid, stream);\n", code.ToString());
        }
        
        [Test]
        public void TestLaunchCudaKernel()
        {
            var builder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            var methodBuilder = new CodeBuilderFactory().CreateCppCUDACodeBuilder();
            
            var block = new Range("block", new Literal("10"), new Literal("1"), new Literal("1"));
            var grid = new Range("grid",  new Literal("3"), new Literal("1"), new Literal("1"));
            var stream = new Stream("stream");

            var func = new Function("func", new VoidType(), new FunctionArguments(), methodBuilder);
            var kernelFunc = new KernelFunction(func, block, grid, stream);

            var code = builder.LaunchGpuKernel(kernelFunc).Build();
            Assert.AreEqual("func<<<grid, block, 0, stream>>>();\n", code.ToString());
        } 
    }
}