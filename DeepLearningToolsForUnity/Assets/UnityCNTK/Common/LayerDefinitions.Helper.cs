using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CNTK;

namespace UnityCNTK.LayerDefinitions
{
    public static class LayerDefineHelper
    {
        public static ResNodeDenseDef[] ResNodeLayers(int numOfNodes, int hiddenSize, NormalizationMethod normalizatoin = NormalizationMethod.None,float dropout = 0.0f)
        {
            var result = new ResNodeDenseDef[numOfNodes];
            for(int i = 0; i < numOfNodes; ++i)
            {
                result[i]=new ResNodeDenseDef(hiddenSize, normalizatoin, dropout);
            }
            return result;
        }

        public static LayerDenseDef[] DenseLayers(int numOfLayers, int hiddenSize, bool hasBias = true, NormalizationMethod normalizatoin = NormalizationMethod.None, float dropout = 0.0f, float initialWeightScale = 0.1f, LayerDef activation = null)
        {
            var result = new LayerDenseDef[numOfLayers];
            for (int i = 0; i < numOfLayers; ++i)
            {
                result[i] = new LayerDenseDef(hiddenSize, normalizatoin, activation, dropout);
                result[i].HasBias = hasBias;
                result[i].InitialWeightScale = initialWeightScale;
            }
            return result;
        }
    }
}