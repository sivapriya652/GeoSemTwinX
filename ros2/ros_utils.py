"""ROS 2 message helpers (no cv_bridge / sensor_msgs_py needed, so they work with any NumPy version)."""
import numpy as np
from scipy.spatial.transform import Rotation


def image_to_numpy(msg):
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8", "rgba8", "bgra8", "mono8", "8uc1", "8uc3"):
        ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8uc1": 1, "8uc3": 3}[enc]
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
        if enc in ("bgr8", "8uc3"):
            a = a[..., ::-1]
        elif enc == "bgra8":
            a = a[..., [2, 1, 0]]
        elif enc == "rgba8":
            a = a[..., :3]
        elif ch == 1:
            a = np.repeat(a, 3, axis=2)
        return np.ascontiguousarray(a)
    if enc in ("16uc1", "mono16"):
        dt = np.dtype(">u2" if msg.is_bigendian else "<u2")
        a = np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // 2)[:, : msg.width]
        return a.astype(np.float32) / 1000.0                         # millimetres -> metres
    if enc == "32fc1":
        dt = np.dtype(">f4" if msg.is_bigendian else "<f4")
        a = np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // 4)[:, : msg.width].astype(np.float32)
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError(f"unsupported image encoding {msg.encoding}")


def numpy_to_image(Image, arr, encoding, stamp=None, frame_id=""):
    m = Image()
    if stamp is not None:
        m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.height, m.width = arr.shape[:2]
    m.encoding = encoding
    m.is_bigendian = 0
    a = np.ascontiguousarray(arr)
    m.step = int(a.strides[0])
    m.data = a.tobytes()
    return m


def matrix_to_pose(Pose, T):
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = float(x), float(y), float(z), float(w)
    return p


def pose_to_matrix(p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]).as_matrix()
    T[:3, 3] = (p.position.x, p.position.y, p.position.z)
    return T


def make_pointcloud2(PointCloud2, PointField, points, colors, stamp, frame_id):
    """XYZ + packed RGB (float32) point cloud."""
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = stamp, frame_id
    n = len(points)
    msg.height, msg.width = 1, n
    msg.fields = [PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                  PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)]
    msg.is_bigendian, msg.point_step, msg.row_step, msg.is_dense = False, 16, 16 * n, True
    c = (np.clip(colors, 0, 1) * 255).astype(np.uint32)
    rgb = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32) if n else np.zeros(0, np.float32)
    buf = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    if n:
        buf["x"], buf["y"], buf["z"], buf["rgb"] = points[:, 0], points[:, 1], points[:, 2], rgb
    msg.data = buf.tobytes()
    return msg


CLASS_RGB = [(0.12, 0.36, 0.85), (0.85, 0.45, 0.10), (0.16, 0.62, 0.50), (0.63, 0.31, 0.69), (0.76, 0.23, 0.35),
             (0.36, 0.50, 0.12), (0.54, 0.43, 0.23), (0.23, 0.56, 0.72)]


def asset_markers(MarkerArray, Marker, assets, classes, stamp, frame_id, ns="geosemtwinx"):
    arr = MarkerArray()
    d = Marker(); d.action = Marker.DELETEALL; d.header.frame_id = frame_id; arr.markers.append(d)
    for a in assets:
        r, g, b = CLASS_RGB[classes.index(a["label"]) % len(CLASS_RGB)] if a["label"] in classes else (0.5, 0.5, 0.5)
        lo, hi = np.asarray(a["lo"]), np.asarray(a["hi"])
        m = Marker(); m.header.stamp, m.header.frame_id = stamp, frame_id
        m.ns, m.id, m.type, m.action = ns + "/box", int(a["id"]), Marker.CUBE, Marker.ADD
        c = (lo + hi) / 2; s = np.maximum(hi - lo, 0.02)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in c)
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in s)
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.35
        arr.markers.append(m)
        t = Marker(); t.header.stamp, t.header.frame_id = stamp, frame_id
        t.ns, t.id, t.type, t.action = ns + "/label", int(a["id"]), Marker.TEXT_VIEW_FACING, Marker.ADD
        t.pose.position.x, t.pose.position.y, t.pose.position.z = float(c[0]), float(c[1]), float(hi[2] + 0.25)
        t.pose.orientation.w = 1.0
        t.scale.z = 0.25
        t.color.r = t.color.g = t.color.b = 0.1; t.color.a = 1.0
        t.text = f"A{int(a['id']):02d} {a['label']} {a['confidence']:.2f}"
        arr.markers.append(t)
    return arr


def traversability_grid(OccupancyGrid, trav, occ, floor_z, stamp, frame_id):
    g = OccupancyGrid()
    g.header.stamp, g.header.frame_id = stamp, frame_id
    g.info.resolution = float(occ.res)
    g.info.width, g.info.height = int(trav.shape[0]), int(trav.shape[1])        # trav is indexed [ix, iy]
    g.info.origin.position.x, g.info.origin.position.y = float(occ.min[0]), float(occ.min[1])
    g.info.origin.position.z = float(floor_z)
    g.info.origin.orientation.w = 1.0
    vals = np.full(trav.shape, -1, np.int8); vals[trav == 1] = 0; vals[trav == 0] = 100
    g.data = vals.T.ravel().astype(np.int8).tolist()                               # row-major, x fastest
    return g


def pack_status(d):
    import json
    return json.dumps(d, default=float)
