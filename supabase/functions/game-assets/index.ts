// Edge Function: game-assets
// Handles storage operations for game assets, restricted to Ivy's account
// Uses service_role key server-side to bypass RLS

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const SUPABASE_URL = Deno.env.get('SUPABASE_URL')!
const SERVICE_ROLE_KEY = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
const ALLOWED_USER_ID = '9735b162-0ee4-4a35-ab48-80ba9ee2ab1f' // Ivy's user ID (savecharlie@gmail.com)
const BUCKET = 'Game assets'
const PATH = 'admin/assets'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
}

Deno.serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Get user from auth header
    const authHeader = req.headers.get('Authorization')
    if (!authHeader) {
      return new Response(JSON.stringify({ error: 'No auth header' }), {
        status: 401,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    // Verify user identity using the admin client with the user's token
    const token = authHeader.replace('Bearer ', '')
    const { data: { user }, error: authError } = await createClient(SUPABASE_URL, SERVICE_ROLE_KEY)
      .auth.getUser(token)

    if (authError || !user || user.id !== ALLOWED_USER_ID) {
      return new Response(JSON.stringify({
        error: 'Unauthorized',
        debug: {
          authError: authError?.message || null,
          hasUser: !!user,
          userId: user?.id || null,
          expectedId: ALLOWED_USER_ID,
          match: user?.id === ALLOWED_USER_ID
        }
      }), {
        status: 403,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    // Create admin client with service_role key
    const adminClient = createClient(SUPABASE_URL, SERVICE_ROLE_KEY)

    const url = new URL(req.url)
    const action = url.searchParams.get('action')

    // LIST assets
    if (req.method === 'GET' && action === 'list') {
      const { data, error } = await adminClient.storage
        .from(BUCKET)
        .list(PATH, { limit: 500 })

      if (error) throw error

      const assets = data
        .filter(f => f.name.endsWith('.png'))
        .map(f => ({ name: f.name, size: f.metadata?.size || 0 }))

      return new Response(JSON.stringify(assets), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    // GET signed URL for asset
    if (req.method === 'GET' && action === 'url') {
      const filename = url.searchParams.get('file')
      if (!filename) {
        return new Response(JSON.stringify({ error: 'No filename' }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        })
      }

      const { data, error } = await adminClient.storage
        .from(BUCKET)
        .createSignedUrl(`${PATH}/${filename}`, 300)

      if (error) throw error

      return new Response(JSON.stringify({ url: data.signedUrl }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    // GET batch signed URLs
    if (req.method === 'GET' && action === 'urls') {
      const files = url.searchParams.get('files')?.split(',') || []
      if (files.length === 0) {
        return new Response(JSON.stringify({ urls: {} }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        })
      }

      const { data, error } = await adminClient.storage
        .from(BUCKET)
        .createSignedUrls(files.map(f => `${PATH}/${f}`), 300)

      if (error) throw error

      const urls: Record<string, string> = {}
      data.forEach(item => {
        if (item.signedUrl) {
          const filename = item.path?.split('/').pop() || ''
          urls[filename] = item.signedUrl
        }
      })

      return new Response(JSON.stringify({ urls }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    // UPLOAD asset
    if (req.method === 'POST') {
      const formData = await req.formData()
      const file = formData.get('file') as File
      const filename = formData.get('filename') as string

      if (!file || !filename) {
        return new Response(JSON.stringify({ error: 'Missing file or filename' }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        })
      }

      const { error } = await adminClient.storage
        .from(BUCKET)
        .upload(`${PATH}/${filename}`, file, {
          contentType: 'image/png',
          upsert: true
        })

      if (error) throw error

      return new Response(JSON.stringify({ success: true, filename }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    // DELETE asset
    if (req.method === 'DELETE') {
      const filename = url.searchParams.get('file')
      if (!filename) {
        return new Response(JSON.stringify({ error: 'No filename' }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        })
      }

      const { error } = await adminClient.storage
        .from(BUCKET)
        .remove([`${PATH}/${filename}`])

      if (error) throw error

      return new Response(JSON.stringify({ success: true }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      })
    }

    return new Response(JSON.stringify({ error: 'Unknown action' }), {
      status: 400,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    })

  } catch (err) {
    return new Response(JSON.stringify({ error: err.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' }
    })
  }
})
