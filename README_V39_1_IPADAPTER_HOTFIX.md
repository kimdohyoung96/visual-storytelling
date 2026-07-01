# V39.1 IP-Adapter hotfix

This fixes the crash:
`encoder_hid_dim_type='ip_image_proj' requires image_embeds`

Cause: once IP-Adapter is loaded globally, diffusers requires an `ip_adapter_image` for every SDXL pipeline call, including background-only generation.

Fix:
- background plate generation now passes a neutral blank image with IP-Adapter scale 0.0
- foreground generation still uses the protagonist reference image
- one-pass fallback calls also pass a neutral image if needed
