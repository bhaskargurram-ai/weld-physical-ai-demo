# Mosaico reference notes (from mosaico-labs/mosaico @ main, SDK 0.8.0)

Source checkout for reading: /home/user/ref/mosaico

## mosaicod
- Image `ghcr.io/mosaico-labs/mosaicod:latest`; gRPC/Arrow Flight on port 6726; only needs Postgres.
- Env: MOSAICOD_DB_URL (no creds), MOSAICOD_DB_USER, MOSAICOD_DB_PASSWORD, MOSAICOD_STORE_ENDPOINT (file:/// or s3 http url), MOSAICOD_STORE_BUCKET.
- command: `server --host 0.0.0.0 --port 6726 --log-level info`
- Build locally: `docker build -f docker/dockerfiles/Dockerfile.mosaicod -t mosaicod:latest mosaicod` (Rust source in mosaicod/)

## SDK
`pip install mosaicolabs` (py>=3.10). `MosaicoClient.connect(host=, port=)` context manager.
Write:
```python
with client.sequence_create(sequence_name=..., metadata={...}, on_error=SessionLevelErrorPolicy.Delete) as sw:
    t = sw.topic_create(topic_name="/x", metadata={}, ontology_type=IMU)
    with t:
        t.push(message=Message(timestamp_ns=ts, data=IMU(...)))
```
Types (top-level mosaicolabs): Image, CompressedImage(data, format: ImageFormat JPEG/PNG/H264/HEVC),
CompressedImage.from_image(pil, format=ImageFormat.JPEG, quality=90), CameraInfo, IMU, RobotJoint(names, positions, velocities, efforts),
Pose(position=Point3d, orientation=Quaternion), Transform, Temperature, Pressure, Floating32/64, String, custom `Serializable` subclass
(see mosaicolabs/examples/bridges/ros/custom_ontology/isaac*.py for custom ontology pattern).

Query:
```python
r = client.query(QuerySequence().with_name_match("SP-*").with_user_metadata("process", eq="FCAW"),
                 QueryTopic().with_name("/imu"), QueryOntologyCatalog(IMU.Q.acceleration.z.gt(15.0)))
for item in r:
    for t in item.topics:
        for c in t.clusterize(int(1e9)): c.timerange.start, c.timerange.end
```
Read: `seq = client.sequence_handler(name)` -> .topics, .user_metadata, .timestamp_ns_min/max;
`seq.get_data_streamer(topics=[...], start_timestamp_ns=, end_timestamp_ns=)` yields (topic, msg); `msg.get_data(IMU)`.
`client.list_sequences()`, `sequence_exists`, `sequence_delete`.

ML: `from mosaicolabs.ml import DataFrameExtractor, SyncTransformer, VideoDecodingTransformer, SyncHold, SyncAsOf`
```python
ext = DataFrameExtractor(seq); dec = VideoDecodingTransformer(topics=["/cam"]); sync = SyncTransformer(target_fps=30.0, policy=SyncHold())
for chunk in ext.to_pandas_chunks(topics=[...], window_sec=5.0, timestamp_ns_start=None, timestamp_ns_end=None):
    dense = sync.transform(dec.transform(chunk))  # columns "{topic}.{Ontology}.{field}"
```
NOTE: docs show `batch_size=` arg to to_pandas_chunks — it does not exist.

Bridges: `mosaico inject ros ./x.mcap --name Run01`, `mosaicolabs.bridges.ros.RosbagInjector`, `bridges/lerobot`.
Examples: mosaicolabs/examples/{query_catalogs.py, bridges/ros/injection.py}; docs/main/docs/learn/*.mdx
