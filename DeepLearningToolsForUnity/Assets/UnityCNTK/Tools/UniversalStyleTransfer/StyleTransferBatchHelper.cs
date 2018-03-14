using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityCNTK;
using UnityCNTK.Tools.StyleAndTexture;
using UnityEngine;


namespace UnityCNTK
{
    [ExecuteInEditMode]
    public class StyleTransferBatchHelper : MonoBehaviour
    {

        //threading stuff
        protected List<Action> _actions = new List<Action>(8);
        protected List<Action> _backlog = new List<Action>(8);
        protected bool _queued = false;



        [SerializeField]
        public MaterialTexture2DDictionary matTextureDic;
        [SerializeField]
        public Texture2DTexture2DDictionary oldNewTextureDic;

        [SerializeField]
        public List<UniversalStyleTransferModel.ParameterSet> styleTransferParams;

        public Texture2D styleTexture;

        //style transfer model
        protected UniversalStyleTransferModel styleTransferModel = null;
        public TextAsset styleTransferModelData;    //the data of the style transfer. need to be set in inspector to the data file


        private void Start()
        {
            //add the parameter sets
            styleTransferParams = new List<UniversalStyleTransferModel.ParameterSet>();
            for (int i = 0; i < 5; ++i)
            {
                styleTransferParams.Insert(0, new UniversalStyleTransferModel.ParameterSet((UniversalStyleTransferModel.PassIndex)i));
            }
        }

        private void Update()
        {
            if (_queued)
            {
                lock (_backlog)
                {
                    var tmp = _actions;
                    _actions = _backlog;
                    _backlog = tmp;
                    _queued = false;
                }

                foreach (var action in _actions)
                    action();

                _actions.Clear();
            }
        }

        public void UpdateMaterialList()
        {
            var rends = FindObjectsOfType<Renderer>();
            matTextureDic = new MaterialTexture2DDictionary();
            oldNewTextureDic = new Texture2DTexture2DDictionary();
            //var rends = GetComponentsInChildren<Renderer>();
            foreach (var r in rends)
            {
                if (r.sharedMaterial != null && r.sharedMaterial.mainTexture != null && r.sharedMaterial.mainTexture is Texture2D)
                {
                    if (!matTextureDic.ContainsKey(r.sharedMaterial))
                        matTextureDic[r.sharedMaterial] = r.sharedMaterial.mainTexture as Texture2D;
                    if (!oldNewTextureDic.ContainsKey(r.sharedMaterial.mainTexture as Texture2D))
                        oldNewTextureDic[r.sharedMaterial.mainTexture as Texture2D] = null;
                }
            }
        }

        public void TransferAll()
        {
            foreach(var t in oldNewTextureDic.Keys)
            {
                TransferOne(t, styleTexture, new Vector2Int(256, 256), new Vector2Int(256, 256), (newT) =>
                {
                    oldNewTextureDic[t] = newT;
                    foreach (var m in matTextureDic)
                    {
                        if (m.Value == t)
                        {
                            m.Key.mainTexture = newT;
                        }
                    }
                });
            }
        }


        protected void TransferOne(Texture2D contentTexture, Texture2D styleTexture, Vector2Int contentResize, Vector2Int styleResize, Action<Texture2D> onFinished)
        {

            styleTransferModel = new UniversalStyleTransferModel(CNTK.DeviceDescriptor.GPUDevice(0), styleTransferModelData.bytes);
            styleTransferModel.CreateModelWithDimensions(contentResize, styleResize);

            var tempContentTexture = Images.GetReadableTextureFromUnreadable(contentTexture);
            byte[] contentBytes = tempContentTexture.GetRGB24FromTexture2D(contentResize);
            

            var tempStyleTexture = Images.GetReadableTextureFromUnreadable(styleTexture);
            byte[] styleBytes = tempStyleTexture.GetRGB24FromTexture2D(styleResize);
            
            DestroyImmediate(tempStyleTexture);
            DestroyImmediate(tempContentTexture);

            Task.Run(() =>
            {
                byte[] result = styleTransferModel.TransferStyle(contentBytes, styleBytes, styleTransferParams.ToArray());
                GC.Collect();
                RunOnMainThread(() => {
                    Texture2D tex = new Texture2D(contentResize.x, contentResize.y, TextureFormat.RGB24, false);
                    tex.LoadRawTextureData(result);
                    tex.Apply();

                    onFinished.Invoke(tex);
                });

            });
        }

        /// <summary>
        /// ccall this in other thread to add action in the main thread.
        /// </summary>
        /// <param name="action"></param>
        public void RunOnMainThread(Action action)
        {
            lock (_backlog)
            {
                _backlog.Add(action);
                _queued = true;
            }
        }


    }
}