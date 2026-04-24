# BENCHMARK.md — Multi-Memory Agent vs No-Memory Agent

> Benchmark chạy lúc: 2026-04-24 12:16
> Model: Qwen2.5:7b (Colab Cloudflare tunnel)
> Env: LangGraph + ChromaDB/Keyword fallback

## Tổng quan

| # | Scenario | Category | No-Memory Result | With-Memory Result | Pass (no-mem) | Pass (with-mem) |
|---|----------|----------|------------------|--------------------|:---:|:---:|
| 1 | Profile recall — tên người dùng | profile_recall | Tôi không nhớ tên bạn vì chúng tôi chưa được giới thiệu với ... | Xin chào! Hiện tại tôi chưa có thông tin về tên của bạn tron... | ❌ | ❌ |
| 2 | Conflict update — dị ứng thực phẩm | conflict_update | Bạn nói mình dị ứng đậu nành rồi. Vậy nên các loại sữa thay ... | Theo như bạn nói trước đó, bạn đang bị dị ứng với đậu nành. ... | ✅ | ✅ |
| 3 | Episodic recall — sự kiện debug Docker | episodic_recall | Xin lỗi, tôi không nhớ được thông tin cụ thể về các hoạt độn... | 上次您提到修复了一个关于使用服务名称替换localhost的问题，具体是在Docker环境中解决的bug。如果您还有其他... | ❌ | ✅ |
| 4 | Semantic retrieval — FAQ về Docker | semantic_retrieval | Khi bạn định nghĩa một service trong file `docker-compose.ym... | Dùng tên dịch vụ (service name) trong Docker giúp bạn truy c... | ❌ | ✅ |
| 5 | Trim / token budget — hội thoại dài | token_budget | Turn 9: Không, bạn chưa giới thiệu tên mình với tôi. Tôi có ... | Xin lỗi, vì chưa có thông tin về bạn trong hồ sơ, nên tôi kh... | ❌ | ❌ |
| 6 | Profile — nghề nghiệp và sở thích | profile_recall | Bạn chưa chia sẻ thông tin về nghề nghiệp của mình, nên tôi ... | Bạn đã chia sẻ rằng mình là bác sĩ. Ngoài ra, tôi cũng biết ... | ❌ | ✅ |
| 7 | Episodic — học LangGraph thành công | episodic_recall | Tôi không nhớ chi tiết các hoạt động cụ thể của bạn, vì tôi ... | Có, bạn vừa chia sẻ rằng bạn đã hoàn thành lab LangGraph. Ch... | ❌ | ✅ |
| 8 | Semantic — giải thích RAG | semantic_retrieval | RAG là viết tắt của Reinforced Active Generation, một phương... | RAG (Retrieval-Augmented Generation) là một kỹ thuật AI mới ... | ❌ | ✅ |
| 9 | Conflict — cập nhật địa chỉ | conflict_update | Bạn đang ở Hà Nội.... | Bạn đang sống ở Hà Nội bây giờ đó! Bạn cần tôi giúp gì tại đ... | ✅ | ✅ |
| 10 | Full memory — kết hợp nhiều loại | full_stack | Tôi không lưu giữ thông tin cá nhân cụ thể của từng người dù... | Theo như tôi hiểu, chúng ta chưa có thông tin về bạn trong h... | ❌ | ❌ |

**No-memory pass rate: 2/10**  
**With-memory pass rate: 7/10**

---

## Chi tiết từng scenario

### Scenario 1: Profile recall — tên người dùng
- **Category:** profile_recall
- **Expected keyword:** `Toàn`

**No-Memory response (eval turn):**
> Tôi không nhớ tên bạn vì chúng tôi chưa được giới thiệu với nhau. Bạn muốn tôi gọi bạn là gì?

**With-Memory response (eval turn):**
> Xin chào! Hiện tại tôi chưa có thông tin về tên của bạn trong hồ sơ. Hãy cho tôi biết tên của bạn nếu bạn muốn tôi ghi nhận và sử dụng trong tương lai nhé!

- No-memory: FAIL ❌
- With-memory: FAIL ❌
- Tokens used (approx): no-mem=1265, with-mem=1057

---

### Scenario 2: Conflict update — dị ứng thực phẩm
- **Category:** conflict_update
- **Expected keyword:** `đậu nành`

**No-Memory response (eval turn):**
> Bạn nói mình dị ứng đậu nành rồi. Vậy nên các loại sữa thay thế tốt cho bạn có thể là sữa hạt từ hạnh nhân, óc chó hoặc hạt điều, cũng như các loại sữa từ gạo, ngô hoặc cây công nghiệp như cà phê hay 

**With-Memory response (eval turn):**
> Theo như bạn nói trước đó, bạn đang bị dị ứng với đậu nành. Bạn đã xác nhận lại rằng mình không dị ứng sữa bò mà chỉ dị ứng đậu nành. Nếu bạn cần thêm thông tin hoặc hỗ trợ về việc thay thế các loại t

- No-memory: PASS ✅
- With-memory: PASS ✅
- Tokens used (approx): no-mem=868, with-mem=918

---

### Scenario 3: Episodic recall — sự kiện debug Docker
- **Category:** episodic_recall
- **Expected keyword:** `docker`

**No-Memory response (eval turn):**
> Xin lỗi, tôi không nhớ được thông tin cụ thể về các hoạt động của bạn trong quá khứ vì tôi là một trợ lý AI và không có khả năng lưu trữ hoặc truy cập vào thông tin cá nhân của bạn. Bạn có thể kiểm tr

**With-Memory response (eval turn):**
> 上次您提到修复了一个关于使用服务名称替换localhost的问题，具体是在Docker环境中解决的bug。如果您还有其他需要帮助的事情或想要分享的信息，请随时告诉我！

- No-memory: FAIL ❌
- With-memory: PASS ✅
- Tokens used (approx): no-mem=590, with-mem=485

---

### Scenario 4: Semantic retrieval — FAQ về Docker
- **Category:** semantic_retrieval
- **Expected keyword:** `service name`

**No-Memory response (eval turn):**
> Khi bạn định nghĩa một service trong file `docker-compose.yml`, bạn có thể sử dụng tên của service này trong môi trường Kubernetes-like hoặc làm host cho các dịch vụ khác. Ví dụ:

```yaml
version: '3'

**With-Memory response (eval turn):**
> Dùng tên dịch vụ (service name) trong Docker giúp bạn truy cập vào các container khác nhau mà không cần biết IP cụ thể của chúng. Đây là một cách tuyệt vời để quản lý kết nối giữa các dịch vụ chạy tro

- No-memory: FAIL ❌
- With-memory: PASS ✅
- Tokens used (approx): no-mem=433, with-mem=1144

---

### Scenario 5: Trim / token budget — hội thoại dài
- **Category:** token_budget
- **Expected keyword:** `Minh`

**No-Memory response (eval turn):**
> Turn 9: Không, bạn chưa giới thiệu tên mình với tôi. Tôi có thể nhớ tên bạn nếu bạn cho biết.

**With-Memory response (eval turn):**
> Xin lỗi, vì chưa có thông tin về bạn trong hồ sơ, nên tôi không nhớ tên của bạn. Nhưng đừng lo, chúng ta có thể gọi bạn là "Người dùng" hoặc sử dụng nick danh khác nếu bạn muốn. Bạn có thể cho tôi biế

- No-memory: FAIL ❌
- With-memory: FAIL ❌
- Tokens used (approx): no-mem=1166, with-mem=1005

---

### Scenario 6: Profile — nghề nghiệp và sở thích
- **Category:** profile_recall
- **Expected keyword:** `bác sĩ`

**No-Memory response (eval turn):**
> Bạn chưa chia sẻ thông tin về nghề nghiệp của mình, nên tôi không biết được. Bạn có thể nói cho tôi biết không?

**With-Memory response (eval turn):**
> Bạn đã chia sẻ rằng mình là bác sĩ. Ngoài ra, tôi cũng biết bạn thích chơi tennis và công việc khá bận rộn. Bạn có muốn chia sẻ thêm về công việc của mình không? Tôi có thể đưa ra một số lời khuyên ho

- No-memory: FAIL ❌
- With-memory: PASS ✅
- Tokens used (approx): no-mem=536, with-mem=584

---

### Scenario 7: Episodic — học LangGraph thành công
- **Category:** episodic_recall
- **Expected keyword:** `langgraph`

**No-Memory response (eval turn):**
> Tôi không nhớ chi tiết các hoạt động cụ thể của bạn, vì tôi không lưu trữ thông tin cá nhân. Nhưng tôi đây để hỗ trợ bạn với bất kỳ thông tin nào bạn muốn chia sẻ!

**With-Memory response (eval turn):**
> Có, bạn vừa chia sẻ rằng bạn đã hoàn thành lab LangGraph. Chúc mừng nhé! Đó là một thành tựu đáng congratulation。如果你有任何其他问题或想要分享的内容，欢迎随时告诉我！

- No-memory: FAIL ❌
- With-memory: PASS ✅
- Tokens used (approx): no-mem=493, with-mem=395

---

### Scenario 8: Semantic — giải thích RAG
- **Category:** semantic_retrieval
- **Expected keyword:** `retrieval`

**No-Memory response (eval turn):**
> RAG là viết tắt của Reinforced Active Generation, một phương pháp sử dụng học tăng cường để tối ưu hóa quá trình tạo văn bản. Nó giúp hệ thống AI ra quyết định về cách thức và nội dung tạo ra văn bản 

**With-Memory response (eval turn):**
> RAG (Retrieval-Augmented Generation) là một kỹ thuật AI mới nổi trong lĩnh vực xử lý ngôn ngữ tự nhiên. Kỹ thuật này kết hợp giữa việc truy vấn dữ liệu từ cơ sở dữ liệu và sinh ra nội dung mới.

Trong

- No-memory: FAIL ❌
- With-memory: PASS ✅
- Tokens used (approx): no-mem=573, with-mem=935

---

### Scenario 9: Conflict — cập nhật địa chỉ
- **Category:** conflict_update
- **Expected keyword:** `hà nội`

**No-Memory response (eval turn):**
> Bạn đang ở Hà Nội.

**With-Memory response (eval turn):**
> Bạn đang sống ở Hà Nội bây giờ đó! Bạn cần tôi giúp gì tại đây? Có thể bạn muốn tìm kiếm địa điểm ăn uống, giải trí hoặc thông tin về các dịch vụ khác tại Hà Nội.

- No-memory: PASS ✅
- With-memory: PASS ✅
- Tokens used (approx): no-mem=414, with-mem=400

---

### Scenario 10: Full memory — kết hợp nhiều loại
- **Category:** full_stack
- **Expected keyword:** `lan`

**No-Memory response (eval turn):**
> Tôi không lưu giữ thông tin cá nhân cụ thể của từng người dùng. Tôi được thiết kế để bảo vệ quyền riêng tư và không lưu trữ dữ liệu cá nhân.

**With-Memory response (eval turn):**
> Theo như tôi hiểu, chúng ta chưa có thông tin về bạn trong hồ sơ hoặc lịch sử hội thoại trước đó. Bạn có thể chia sẻ thêm để tôi giúp đỡ tốt hơn không? Ví dụ, nếu bạn muốn tôi gọi bạn bằng tên nào, ho

- No-memory: FAIL ❌
- With-memory: FAIL ❌
- Tokens used (approx): no-mem=862, with-mem=804

---

## Phân tích theo category

| Category | Scenarios | With-Memory Pass |
|----------|-----------|-----------------|
| profile_recall | 2 | 1/2 |
| conflict_update | 2 | 2/2 |
| episodic_recall | 2 | 2/2 |
| semantic_retrieval | 2 | 2/2 |
| token_budget | 1 | 0/1 |
| full_stack | 1 | 0/1 |

---

## Reflection — Privacy & Limitations

### 1. Memory nào giúp agent nhất?
**Short-term memory** đóng vai trò quan trọng nhất trong mỗi turn, giúp agent nhớ context hội thoại gần đây.
**Long-term profile** giúp agent nhớ thông tin người dùng xuyên suốt session.

### 2. Memory nào rủi ro nhất nếu retrieve sai?
**Long-term profile** nguy hiểm nhất. Nếu extract sai facts (ví dụ: nhầm dị ứng sữa bò → lưu thành đậu nành trước khi user sửa),
agent có thể đưa ra lời khuyên sức khỏe sai lệch. Đặc biệt nguy hiểm với thông tin y tế, tài chính.

### 3. Nếu user yêu cầu xóa memory, xóa ở backend nào?
- **Short-term**: Clear `ConversationBufferMemory` — dễ, chỉ cần reset deque.
- **Long-term profile**: Xóa file `data/user_profile.json` hoặc gọi `UserProfileMemory.clear()`.
- **Episodic**: Xóa file `data/episodic_log.json` hoặc filter theo user_id.
- **Semantic**: Clear ChromaDB collection — cần xóa toàn bộ nếu không có user partitioning.

> ⚠️ **PII Risk**: Long-term profile lưu tên, nghề nghiệp, dị ứng — thông tin nhạy cảm.
> Cần có consent rõ ràng, TTL policy, và right-to-delete trước khi deploy production.

### 4. Điều gì sẽ làm system fail khi scale?
- **ChromaDB single-node**: không hỗ trợ multi-user isolation tốt, cần namespace per user.
- **JSON file store** cho profile/episodic: không phù hợp concurrent access, cần migrate lên database.
- **No TTL**: Episodic memory tích lũy không giới hạn → context window bị vượt nếu không trim.
- **LLM extraction (extract_facts)**: Có thể extract sai không có human review, cần confidence threshold.
- **Token budget tĩnh**: Budget 2000 tokens phù hợp Qwen2.5:7b nhưng cần điều chỉnh cho model khác.

### 5. Limitations kỹ thuật hiện tại
- Memory không có user_id partitioning → không thể multi-user.
- Semantic embeddings dùng ChromaDB default model (all-minilm), không fine-tuned cho tiếng Việt.
- Conflict resolution đơn giản (overwrite) — không có merge/diff cho structured data.
- Không có async support → mỗi turn phải chờ LLM response xong.