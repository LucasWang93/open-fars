# Advantage-Guided Curriculum Learning for Web Navigation

**Hypothesis:** Introducing advantage-guided curriculum learning will improve step_success_rate by prioritizing training on simpler tasks first and gradually increasing complexity, leveraging advantage estimation to optimize action planning.

**Key Innovation:** Combining curriculum learning with advantage estimation to sequence tasks based on complexity and predicted utility, while using visual and textual webpage representations for richer context.

## Method
- Training Strategy: curriculum_learning
- Data Processing: html_plus_screenshot_caption
- Prompt Design: reflection
- Model Config: lora_r32
- Augmentation: trajectory_augmentation

**Rationale:** Advantage-guided curriculum learning draws from the pattern of on-the-fly advantage estimation (pat_fd09f3128e0d), which has shown to improve action prediction accuracy in web navigation tasks. By sequencing tasks from simple to complex, the model can gradually build task-specific competencies while optimizing action planning using advantage estimation. Additionally, the inclusion of screenshot captions alongside HTML provides richer multimodal context, which enhances understanding of webpage state.
**Novelty:** This approach differs from the baseline vanilla SFT by introducing a curriculum strategy guided by advantage estimation and leveraging multimodal inputs (HTML plus screenshot captions), which were not considered in past experiments.