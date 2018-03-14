using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;
namespace UnityCNTK
{
    public static class CNTKUnityExtensionMethods
    {
        /// <summary>
        /// Shallow copy a IList and retrn a array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="iList"></param>
        /// <param name="startIndex"></param>
        /// <returns></returns>
        public static T[] CopyToArray<T>(this IList<T> iList, int startIndex = 0)
        {
            T[] result = new T[iList.Count - startIndex];
            iList.CopyTo(result, startIndex);
            return result;
        }


        public static Parameter FindParameterByName(this Function func, string name)
        {
            var allInputs = func.Parameters();
            foreach (var p in allInputs)
            {
                if (p.Kind == VariableKind.Parameter)
                {
                    //look for the parameter of the same name from the other function
                    if (name == p.Name) {
                        return p;
                    }
                }
            }
            return null;
        }

        public static List<Parameter> FindParametersByName(this Function func, string[] names)
        {
            HashSet<string> set = new HashSet<string>(names);
            var allInputs = func.Parameters();
            var result = new List<Parameter>();
            foreach (var p in allInputs)
            {
                if (p.Kind == VariableKind.Parameter)
                {
                    //look for the parameter of the same name from the other function
                    if (set.Contains( p.Name))
                    {
                        result.Add(p);
                    }
                }
            }
            return result;
        }

        public static void RestoreParametersByName(this Function func, Function fromFunction)
        {
            var allInputs = func.Parameters();
            var fromInputs = fromFunction.Parameters();
            foreach(var p in allInputs)
            {
                if(p.Kind == VariableKind.Parameter)
                {
                    Parameter fromP = null;
                    //look for the parameter of the same name from the other function
                    foreach(var f in fromInputs)
                    {
                        if(f.Name == p.Name)
                        {
                            fromP = f;
                            break;
                        }
                    }

                    if(fromP == null)
                    {
                        Debug.LogWarning("Did not find parameter " + p.Name +  " in the other function");
                    }else if (!p.Shape.Equals(fromP.Shape))
                    {
                        Debug.LogError("Parameter shapes not the same. Original " + p.Name + " has a shape of " + string.Join(",", p.Shape.Dimensions) + " while the new one is " + string.Join(",", fromP.Shape.Dimensions));
                    }
                    else
                    {
                        p.Value().CopyFrom(fromP.Value());
                    }

                }
            }
        }
    }
}