using UnityEngine;
using UnityEditor;

using UnityCNTK.Tools.StyleAndTexture;
using System.Collections.Generic;
using System.Threading.Tasks;
using CNTK;
using System;

namespace UnityCNTK.Editor
{
    public class StyleTransferWindow : EditorWindow
    {
        
        protected List<Action> _actions = new List<Action>(8);
        protected List<Action> _backlog = new List<Action>(8);
        protected bool _queued = false;


        protected bool IsRunningTransfer { get { return isRunningTransfer; } set
            {
                isRunningTransfer = value;
                contentWindow.enableButtons = !isRunningTransfer;
                styleWindow.enableButtons = !isRunningTransfer;
            }
        }
        protected bool isRunningTransfer = false;

        public Material showImageMat;//will be set in editor
        protected InputImageWindow contentWindow;
        protected InputImageWindow styleWindow;
        protected OutputImageWindow resultWindow;

        //values to use
        protected bool blendingFactorFoldout = false;
        protected List<UniversalStyleTransferModel.ParameterSet> styleTransferParams;

        protected Vector2Int contentSize;
        protected Vector2Int styleSize;

        protected bool resizeContent = true;
        protected bool resizeStyle = true;

        protected bool useOriginalAlpha = true;

        //dimensions for the layout

        protected static Rect windowDefaultRect = new Rect(30, 60, 960, 720);
        protected static float leftColumnMaxWidth = 230;
        protected static Rect contentWindowDefaultRect = new Rect(leftColumnMaxWidth + 40, 30, 300, 330);
        protected static Rect styleWindowDefaultRect = new Rect(leftColumnMaxWidth + 40, 370, 300, 330);
        protected static Rect resultWindowDefaultRect = new Rect(leftColumnMaxWidth + 360, 200, 350, 350);

        //style transfer model
        protected UniversalStyleTransferModel styleTransferModel = null;
        public TextAsset styleTransferModelData;    //the data of the style transfer. need to be set in inspector to the data file

        protected DeviceDescriptor cpuDevice = null;
        protected DeviceDescriptor gpuDevice = null;

        //style related
        protected GUIStyle titleStyle;

        private void Awake()
        {
            contentWindow = new InputImageWindow(1, contentWindowDefaultRect, "Content Image");
            contentWindow.showImageMat = showImageMat;
            styleWindow = new InputImageWindow(2, styleWindowDefaultRect, "Style Image");
            styleWindow.showImageMat = showImageMat;
            resultWindow = new OutputImageWindow(3, resultWindowDefaultRect, "Result Image");
            resultWindow.showImageMat = showImageMat;

            contentSize = new Vector2Int(512, 512);
            styleSize = new Vector2Int(512, 512);

            //add the parameter sets
            styleTransferParams = new List<UniversalStyleTransferModel.ParameterSet>();
            for (int i = 0; i < 5; ++i)
            {
                styleTransferParams.Insert(0,new UniversalStyleTransferModel.ParameterSet((UniversalStyleTransferModel.PassIndex)i));
            }

            //get available device to run the style transfer on
            foreach(var d in DeviceDescriptor.AllDevices())
            {
                if (d.Type == DeviceKind.CPU && cpuDevice == null)
                {
                    cpuDevice = d;
                } else if (d.Type == DeviceKind.GPU && gpuDevice == null)
                {
                    gpuDevice = d;
                }
            }

            
        }

        // Add menu named "My Window" to the Window menu
        [MenuItem("Window/UnityDeeplearningTools/StyleTransferTool")]
        static void Init()
        {
            // Get existing open window or if none, make a new one:
            StyleTransferWindow window = (StyleTransferWindow)EditorWindow.GetWindow(typeof(StyleTransferWindow));
            window.position = windowDefaultRect;
            window.Show();
            

        }

        void OnGUI()
        {


            EditorGUILayout.BeginHorizontal(GUILayout.MaxWidth(leftColumnMaxWidth));
            EditorGUILayout.BeginVertical();

            EditorGUILayout.Space();
            EditorGUILayout.Space();
            EditorGUILayout.Space();
            EditorGUILayout.Space();
            //title
            if (titleStyle == null)
            {
                titleStyle = GUI.skin.GetStyle("IN Title");
            }
            EditorGUILayout.LabelField("Style Transfer", titleStyle);
            EditorGUILayout.HelpBox("Transfer the content image towards the style image.", MessageType.Info);
            
            //images dimensions
            //content
            EditorGUILayout.BeginHorizontal(GUILayout.MaxWidth(leftColumnMaxWidth));
            resizeContent = EditorGUILayout.Toggle(resizeContent);
            EditorGUI.BeginDisabledGroup(!resizeContent);
            contentSize = Vector2Int.Max(EditorGUILayout.Vector2IntField(new GUIContent("Content Image Resize", "Result image will be the same size."), contentSize), new Vector2Int(32, 32));
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.EndHorizontal();
            //style
            EditorGUILayout.BeginHorizontal(GUILayout.MaxWidth(leftColumnMaxWidth));
            resizeStyle = EditorGUILayout.Toggle(resizeStyle);
            EditorGUI.BeginDisabledGroup(!resizeStyle);
            styleSize = Vector2Int.Max(EditorGUILayout.Vector2IntField("Style Image Resize", styleSize), new Vector2Int(32, 32));
            EditorGUI.EndDisabledGroup();
            EditorGUILayout.EndHorizontal();

            if (!resizeContent && contentWindow.TextureOriginalSize != Vector2Int.zero)
                contentSize = contentWindow.TextureOriginalSize;
            if (!resizeStyle && styleWindow.TextureOriginalSize != Vector2Int.zero)
                styleSize = styleWindow.TextureOriginalSize;
            contentWindow.desiredSize = contentSize;
            styleWindow.desiredSize = styleSize;

            //other settings
            EditorGUILayout.BeginHorizontal(GUILayout.MaxWidth(leftColumnMaxWidth));
            useOriginalAlpha = EditorGUILayout.Toggle(useOriginalAlpha);
            EditorGUILayout.LabelField("Use Alpha From Content");
            EditorGUILayout.EndHorizontal();

            //first the blending parameters
            blendingFactorFoldout = EditorGUILayout.Foldout(blendingFactorFoldout, new GUIContent("Blend Strength", "Strength of blending towards style image."));
            if (!blendingFactorFoldout)
            {
                float tempWidth = EditorGUIUtility.labelWidth;
                EditorGUIUtility.labelWidth = 50;
                for (int i = 0; i < styleTransferParams.Count; ++i)
                {
                    EditorGUILayout.BeginHorizontal(GUILayout.MaxWidth(leftColumnMaxWidth));
                    styleTransferParams[i].enabled = EditorGUILayout.Toggle(styleTransferParams[i].enabled);
                    EditorGUI.BeginDisabledGroup(!styleTransferParams[i].enabled);
                    styleTransferParams[i].BlendFactor = EditorGUILayout.Slider("Level " + (5 - i), styleTransferParams[i].BlendFactor, 0, 1);
                    EditorGUI.EndDisabledGroup();
                    EditorGUILayout.EndHorizontal();
                }
                EditorGUIUtility.labelWidth = tempWidth;
            }
            //show warning meesage if no GPU device availabe
            if(gpuDevice == null)
            {
                EditorGUILayout.HelpBox("No available GPU device. Use CPU instead, which might take a minute or more to run.", MessageType.Warning);
            }
            //transfer button
            EditorGUI.BeginDisabledGroup(isRunningTransfer || contentWindow.showTexture == null || styleWindow.showTexture == null);
            if (GUILayout.Button("Transfer"))
            {
                StartTransfer();
            }
            EditorGUI.EndDisabledGroup();

            //help box for the texture sythesizing
            EditorGUILayout.LabelField("Texture Synthesize", titleStyle);
            EditorGUILayout.HelpBox("Set content image to white noise and style image to desired texture for texxture synthesizing",MessageType.Info);
            
            EditorGUILayout.EndVertical();
            EditorGUILayout.EndHorizontal();

            BeginWindows();

            //All GUI.Window or GUILayout.Window must come inside here
            contentWindow.OnGUI();
            styleWindow.OnGUI();
            resultWindow.OnGUI();
            EndWindows();


        }

        void OnInspectorUpdate()
        {
            Repaint();
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

        protected void StartTransfer()
        {
            IsRunningTransfer = true;
            if (gpuDevice != null)
                styleTransferModel = new UniversalStyleTransferModel(gpuDevice, styleTransferModelData.bytes);
            else if (cpuDevice != null)
                styleTransferModel = new UniversalStyleTransferModel(cpuDevice, styleTransferModelData.bytes);
            else
                Debug.LogError("There must be a bug that no device if found");
            styleTransferModel.CreateModelWithDimensions(contentSize, styleSize);
            //var tempContentTexture = Images.GetReadableTextureFromUnreadable(contentWindow.showTexture);
            byte[] contentBytes = contentWindow.showTexture.GetRGB24FromTexture2D(contentSize);
            //contentBytes = contentWindow.showTexture.GetRawTextureData();
            //DestroyImmediate(tempContentTexture);
            //var tempStyleTexture = Images.GetReadableTextureFromUnreadable(styleWindow.showTexture);
            byte[] styleBytes = styleWindow.showTexture.GetRGB24FromTexture2D(styleSize);
            //DestroyImmediate(tempStyleTexture);

            /*
            //run in main thread
            byte[] result = styleTransferModel.TransferStyle(contentBytes, styleBytes, styleTransferParams.ToArray());
            Texture2D tex2 = new Texture2D(contentSize.x, contentSize.y, TextureFormat.RGB24, false);
            tex2.LoadRawTextureData(result);
            tex2.Apply();
            if (resultWindow.showTexture != null)
            {
                DestroyImmediate(resultWindow.showTexture);
            }
            resultWindow.showTexture = tex2;*/

            //run in ahother thread
            Task.Run(() =>
            {
                try
                {
                    byte[] result = styleTransferModel.TransferStyle(contentBytes, styleBytes, styleTransferParams.ToArray());
                    GC.Collect();
                    RunOnMainThread(() => {
                        Texture2D tex = new Texture2D(contentSize.x, contentSize.y, TextureFormat.RGB24, false);
                        tex.LoadRawTextureData(result);
                        tex.Apply();

                        var resultTexture = tex;
                        if (useOriginalAlpha)
                        {
                            resultTexture = Images.GetTextureWithAlpha(tex, contentWindow.showTexture);
                            DestroyImmediate(tex);
                        }

                        if (resultWindow.showTexture != null)
                        {
                            DestroyImmediate(resultWindow.showTexture);
                        }
                        resultWindow.showTexture = resultTexture;
                        IsRunningTransfer = false;
                    });
                }
                catch(Exception e) {
                    Debug.LogError(e.Message);
                     RunOnMainThread(() => {
                         IsRunningTransfer = false;
                     });
                }

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


    public class InputImageWindow : ResizableSubWindow
    {


        public Texture2D showTexture = null;
        public Material showImageMat;

        protected int bottomRowHeight = 20;
        protected int padding = 5;
        protected int bottomButtonWidth = 80;
        
        public Vector2Int desiredSize = new Vector2Int(256,256);
        public Vector2Int TextureOriginalSize { get { return showTexture == null ? Vector2Int.zero : new Vector2Int(showTexture.width, showTexture.height); } }

        public bool enableButtons = true;

        public InputImageWindow(int id, string title = "") : base(id, title)
        {
        }

        public InputImageWindow(int id, Rect initWindowRect, string title = "") : base(id, initWindowRect, title)
        {
        }


        protected override void DoWindow(int windowID)
        {
            Rect imageRect = new Rect(padding, dragableTopHeight, WindowRect.width - padding * 2, WindowRect.height - bottomRowHeight - dragableTopHeight - padding * 2);
            if (showTexture)
                EditorGUI.DrawPreviewTexture(imageRect, showTexture, showImageMat, ScaleMode.ScaleToFit);
            else
                EditorGUI.DrawPreviewTexture(imageRect, EditorGUIUtility.whiteTexture, showImageMat);

            //texture picker
            EditorGUI.BeginDisabledGroup(!enableButtons);
            var currentPickerWindow = EditorGUIUtility.GetControlID(FocusType.Passive) + 100;
            if (GUI.Button(new Rect(padding, WindowRect.height - bottomRowHeight - padding, bottomButtonWidth, bottomRowHeight), "Texture"))
            {

                EditorGUIUtility.ShowObjectPicker<Texture2D>(null, true, "", currentPickerWindow);
            }
            if (Event.current.commandName == "ObjectSelectorClosed" && EditorGUIUtility.GetObjectPickerControlID() == currentPickerWindow)
            {
                if (showTexture != null)
                    GameObject.DestroyImmediate(showTexture);
                var newTex = (Texture2D)EditorGUIUtility.GetObjectPickerObject();
                if (newTex != null)
                {
                    showTexture = Images.GetReadableTextureFromUnreadable(newTex);//always switch it to readable texture first
                }
                else
                {
                    showTexture = null;
                }
                currentPickerWindow = -1;
            }

            //use whitenoise
            if (GUI.Button(new Rect(padding * 2 + bottomButtonWidth, WindowRect.height - bottomRowHeight - padding, bottomButtonWidth, bottomRowHeight), "Whitenoise"))
            {
                GenerateWhiteNoiseTexture();
            }

            EditorGUI.EndDisabledGroup();

            //show original dimension
            string imageInfo = showTexture != null ? "size:" + showTexture.width + "x" + showTexture.height : "";
            EditorGUI.LabelField(new Rect(padding * 3 + bottomButtonWidth * 2, WindowRect.height - bottomRowHeight - padding, bottomButtonWidth*1.5f, bottomRowHeight), imageInfo);

            base.DoWindow(windowID);
        }

        protected void GenerateWhiteNoiseTexture()
        {
            showTexture = new Texture2D(desiredSize.x, desiredSize.y,TextureFormat.RGB24,false);
            //showTexture.SetPixels(Images.GeneratePureColor(desiredSize.x* desiredSize.y,Color.white),0);
            showTexture.SetPixels(Images.GenerateWhiteNoise(desiredSize.x * desiredSize.y), 0);
            showTexture.Apply();
        }
    }


    public class OutputImageWindow : ResizableSubWindow
    {


        public Texture2D showTexture = null;
        public Material showImageMat;

        protected int bottomRowHeight = 20;
        protected int padding = 5;
        protected int bottomButtonWidth = 80;

        public OutputImageWindow(int id, string title = "") : base(id, title)
        {
        }

        public OutputImageWindow(int id, Rect initWindowRect, string title = "") : base(id, initWindowRect, title)
        {
        }


        protected override void DoWindow(int windowID)
        {
            Rect imageRect = new Rect(padding, dragableTopHeight, WindowRect.width - padding * 2, WindowRect.height - bottomRowHeight - dragableTopHeight - padding * 2);
            if (showTexture)
                EditorGUI.DrawPreviewTexture(imageRect, showTexture, showImageMat, ScaleMode.ScaleToFit);
            else
                EditorGUI.DrawPreviewTexture(imageRect, EditorGUIUtility.whiteTexture, showImageMat);

            //Save
             EditorGUI.BeginDisabledGroup(showTexture == null);
             if (GUI.Button(new Rect(padding, WindowRect.height - bottomRowHeight - padding, bottomButtonWidth*1.5f, bottomRowHeight), "Save"))
             {
                 EditorUtils.SaveTextureToPNGFile(showTexture);
             }
             EditorGUI.EndDisabledGroup();

            //show original dimension
            string imageInfo = showTexture != null ? "size:" + showTexture.width + "x" + showTexture.height : "";
            EditorGUI.LabelField(new Rect(padding * 3 + bottomButtonWidth * 2, WindowRect.height - bottomRowHeight - padding, bottomButtonWidth * 1.5f, bottomRowHeight), imageInfo);

            base.DoWindow(windowID);
        }
    }
}