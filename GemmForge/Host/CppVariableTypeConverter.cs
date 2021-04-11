using System;
using GemmForge.Common;

namespace GemmForge.Host
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