from lightllm.server.build_prompt import _flatten_multimodal_content


def test_flatten_text_only_content_parts_matches_plain_text():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    _flatten_multimodal_content(messages)

    assert messages[0]["content"] == "hello"


def test_flatten_preserves_multimodal_part_order():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "compare"},
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AA=="}},
            ],
        }
    ]

    _flatten_multimodal_content(messages)

    assert messages[0]["content"] == "<image>\ncompare\n<audio>"


def test_flatten_leaves_string_content_unchanged():
    messages = [{"role": "system", "content": "keep exactly"}]

    _flatten_multimodal_content(messages)

    assert messages[0]["content"] == "keep exactly"
