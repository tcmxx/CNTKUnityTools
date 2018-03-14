using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace UnityCNTK
{

    [CustomEditor(typeof(StyleTransferBatchHelper))]
    public class StyleTransferBatchInspector : UnityEditor.Editor
    {
        StyleTransferBatchHelper transferHelper;

        private void OnEnable()
        {
            transferHelper = (StyleTransferBatchHelper)target;
        }

        public override void OnInspectorGUI()
        {

            Undo.RecordObject(transferHelper, "StyleTransfer");


            if(GUILayout.Button("Update Materials"))
            {
                transferHelper.UpdateMaterialList();
            }

            //effected materials
            EditorGUILayout.LabelField("Effected materials");
            List<Material> keys = new List<Material>(transferHelper.matTextureDic.Keys);
            foreach(var m in keys)
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.ObjectField(m, typeof(Material),true);
                if (GUILayout.Button("Remove"))
                {
                    transferHelper.matTextureDic.Remove(m);
                }
                EditorGUILayout.EndHorizontal();
            }

            //for test, show the texutre dics
            /*EditorGUILayout.LabelField("Texture old new dic");
            List<Texture2D> texKeys = new List<Texture2D>(transferHelper.oldNewTextureDic.Keys);
            foreach (var t in texKeys)
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.ObjectField(t, typeof(Texture2D), true);
                EditorGUILayout.EndHorizontal();
            }*/
            
            transferHelper.styleTexture = EditorGUILayout.ObjectField("Style Texture",transferHelper.styleTexture, typeof(Texture2D), true) as Texture2D;
            transferHelper.styleTransferModelData = EditorGUILayout.ObjectField("NN data",transferHelper.styleTransferModelData, typeof(TextAsset), true) as TextAsset;

            //transfor parameters
            EditorGUILayout.LabelField("Transfer parameters");
            foreach (var t in transferHelper.styleTransferParams)
            {
                t.BlendFactor = EditorGUILayout.Slider("Blend Strength", t.BlendFactor,0,1);
            }

            if (GUILayout.Button("Transfer All"))
            {
                transferHelper.TransferAll();
            }
        }
    }

}