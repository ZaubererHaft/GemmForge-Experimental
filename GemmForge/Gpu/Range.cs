namespace GemmForge.Common
{
    public class Range 
    {
        public string Name { get; }
        public Expression X { get; }
        public Expression Y { get; }
        public Expression Z { get; }

        public Range(string name, Expression x, Expression y, Expression z)
        {
            Name = name;    
            X = x;
            Y = y;
            Z = z;
        }
    }
}