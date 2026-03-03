export function onRequest() {
  const data = [{
    "relation": ["delegate_permission/common.handle_all_urls"],
    "target": {
      "namespace": "android_app",
      "package_name": "io.locomot.sediment",
      "sha256_cert_fingerprints": [
        "63:51:B5:32:FE:38:CA:BA:14:36:5E:69:13:B0:09:0B:08:C5:C2:25:0B:4A:D9:00:1A:F8:BD:DB:EE:04:7C:1A"
      ]
    }
  }];

  return new Response(JSON.stringify(data), {
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*"
    }
  });
}
