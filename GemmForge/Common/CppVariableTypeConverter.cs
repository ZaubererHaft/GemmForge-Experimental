using System;

namespace GemmForge.Common
{
    public class CppVariableTypeConverter
    {
        public string Convert(VariableType variableType)
        {
            switch (variableType)
            {
                case VariableType.DoublePrecisionFloat:
                    return "double";
                case VariableType.SinglePrecisionFloat:
                    return "float";
            }

            throw new ArgumentException();
        }
    }
}