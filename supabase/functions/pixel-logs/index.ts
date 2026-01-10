// Edge Function: pixel-logs
// Remote console logging for pixel editor debugging

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
}

// In-memory log storage (persists until function cold starts)
const logs: { timestamp: string; level: string; message: string }[] = []
const MAX_LOGS = 200

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const url = new URL(req.url)

    // POST - add log entry
    if (req.method === 'POST') {
      const body = await req.json()
      const entry = {
        timestamp: new Date().toISOString(),
        level: body.level || 'log',
        message: typeof body.message === 'string' ? body.message : JSON.stringify(body.message)
      }
      logs.push(entry)

      // Trim old logs
      while (logs.length > MAX_LOGS) {
        logs.shift()
      }

      return new Response(JSON.stringify({ success: true }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    // GET - retrieve logs
    if (req.method === 'GET') {
      const limit = parseInt(url.searchParams.get('limit') || '100')
      const recentLogs = logs.slice(-limit)
      return new Response(JSON.stringify(recentLogs), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    // DELETE - clear logs
    if (req.method === 'DELETE') {
      logs.length = 0
      return new Response(JSON.stringify({ success: true, cleared: true }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    })

  } catch (err) {
    return new Response(JSON.stringify({ error: err.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    })
  }
})
