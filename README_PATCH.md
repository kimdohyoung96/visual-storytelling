# DCE-ViStory Final Patch

## What this patch fixes

1. Prevents unintended woodcutter/lumberjack/axe leakage.
2. Forces the planner to use only input-grounded entities.
3. Generates images autoregressively:
   - frame 1 = story sentence 1 + input/reference image
   - frame 2 = story sentence 2 + frame 1 + initial anchor
   - frame k = story sentence k + frame k-1 + initial anchor
4. Saves every frame packet so you can inspect exactly what prompt generated each image.

## Files

- `guards.py`: entity constraints and forbidden-term validator.
- `planner.py`: input-grounded DCEE planner and storyboard builder.
- `sequential_pipeline.py`: final sequential frame generation loop.
- `sdxl_image_generator.py`: optional SDXL wrapper.
- `example_panda_input.json`: clean panda-only input example.

## Integration

Copy files into your project:

```text
src/dce_vistory/guards.py
src/dce_vistory/planner.py
src/dce_vistory/sequential_pipeline.py
src/dce_vistory/sdxl_image_generator.py
examples/panda_no_woodcutter.json
```

Then connect your existing LLM function as `llm_json(messages)` and your SDXL/IP-Adapter generator as `image_generator.generate_frame(packet)`.

## Recommended search before running

PowerShell:

```powershell
Select-String -Path .\src\**\*.py,.\configs\**\*,.\examples\**\*,.\prompts\**\* -Pattern "woodcutter","lumberjack","logger","woodsman","axe","나무꾼","도끼"
```

Remove those terms from examples, default prompts, config templates, and few-shot demonstrations unless the current input explicitly allows them.
