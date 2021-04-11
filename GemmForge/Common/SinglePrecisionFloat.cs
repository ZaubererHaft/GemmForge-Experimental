namespace GemmForge.Common
{
    public class SinglePrecisionFloat : VariableType
    {
        public override void Resolve(IVariableResolver resolver)
        {
            resolver.Resolve(this);
        }

        public override string Type => "float";
    }
}