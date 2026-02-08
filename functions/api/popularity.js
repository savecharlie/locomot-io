// Cloudflare Pages Function — queries zone analytics for page popularity
// Env secrets: CF_API_TOKEN, CF_ZONE_ID (set via wrangler pages secret put)

export async function onRequestGet(context) {
  const headers = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': 'https://locomot.io',
    'Cache-Control': 'public, max-age=3600', // cache 1 hour
  };

  const token = context.env.CF_API_TOKEN;
  const zoneId = context.env.CF_ZONE_ID;

  if (!token || !zoneId) {
    return new Response(JSON.stringify({ error: 'missing config' }), { status: 500, headers });
  }

  // Last 7 days (Cloudflare adaptive groups max ~8 days)
  const now = new Date();
  const since = new Date(now - 7 * 24 * 60 * 60 * 1000);
  const sinceStr = since.toISOString().split('T')[0];
  const untilStr = now.toISOString().split('T')[0];

  const query = `query {
    viewer {
      zones(filter: { zoneTag: "${zoneId}" }) {
        httpRequestsAdaptiveGroups(
          filter: {
            date_geq: "${sinceStr}"
            date_leq: "${untilStr}"
          }
          limit: 50
          orderBy: [count_DESC]
        ) {
          count
          dimensions {
            clientRequestPath
          }
        }
      }
    }
  }`;

  try {
    const resp = await fetch('https://api.cloudflare.com/client/v4/graphql', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });

    const data = await resp.json();

    if (data.errors) {
      return new Response(JSON.stringify({ error: data.errors }), { status: 502, headers });
    }

    const groups = data.data?.viewer?.zones?.[0]?.httpRequestsAdaptiveGroups || [];

    // Aggregate by normalized game path
    const counts = {};
    for (const g of groups) {
      const path = g.dimensions.clientRequestPath;
      const count = g.count;

      // Normalize: /sediment/ and /sediment/index.html both count for sediment
      // Match game paths from homepage hrefs
      const gamePatterns = [
        { pattern: /^\/trains(\.html)?$/, key: 'trains.html' },
        { pattern: /^\/ice_skater/, key: 'ice_skater/' },
        { pattern: /^\/pixel\.html/, key: 'pixel.html' },
        { pattern: /^\/painter/, key: 'painter/' },
        { pattern: /^\/sandbox/, key: 'sandbox/' },
        { pattern: /^\/sediment/, key: 'sediment/' },
      ];

      for (const { pattern, key } of gamePatterns) {
        if (pattern.test(path)) {
          counts[key] = (counts[key] || 0) + count;
          break;
        }
      }
    }

    return new Response(JSON.stringify(counts), { status: 200, headers });
  } catch (err) {
    return new Response(JSON.stringify({ error: err.message }), { status: 500, headers });
  }
}
