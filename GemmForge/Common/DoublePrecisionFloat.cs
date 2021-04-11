namespace GemmForge.Common
{
    public class DoublePrecisionFloat : VariableType
    {
        public override void Resolve(IVariableResolver resolver)
        {
            resolver.Resolve(this);
        }

        public override string Type => "double";
    }
}