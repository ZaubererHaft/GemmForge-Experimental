using GemmForge.Common;

namespace GemmForge.Gpu
{
    public class SharedVariableType : VariableType
    {
        private readonly VariableType _subType;

        public SharedVariableType(VariableType subType)
        {
            _subType = subType;
        }

        public override void Resolve(IVariableResolver resolver)
        {
            resolver.Resolve(this);
        }

        public override string Type => _subType.Type;
    }
}