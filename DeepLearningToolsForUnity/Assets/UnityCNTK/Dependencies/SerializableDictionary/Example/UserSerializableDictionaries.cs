using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;

[Serializable]
public class StringStringDictionary : SerializableDictionary<string, string> {}

[Serializable]
public class ObjectColorDictionary : SerializableDictionary<UnityEngine.Object, Color> {}

[Serializable]
public class DicGameobjectFloat : SerializableDictionary<GameObject, float> { }

[Serializable]
public class MaterialTexture2DDictionary : SerializableDictionary<Material, Texture2D> { }
[Serializable]
public class Texture2DTexture2DDictionary : SerializableDictionary<Texture2D, Texture2D> { }