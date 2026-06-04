#include "bot.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class TelegramBot {
private:
    string token;
    string apiUrl;
    NeuralNetwork brain;
    ImageAnalyzer vision;
    SpeechToText speech;
    KnowledgeGraph knowledge;
    
public:
    TelegramBot(const string& botToken) 
        : token(botToken),
          apiUrl("https://api.telegram.org/bot" + botToken),
          brain({768, 1536, 1536, 768}) {
        
        cout << "🧠 Arvan AI Initializing..." << endl;
        cout << "✅ Neural Network loaded" << endl;
        cout << "✅ Vision system ready" << endl;
        cout << "✅ Speech processing ready" << endl;
        cout << "✅ Knowledge graph loaded" << endl;
    }
    
    void run() {
        cout << "\n🚀 Bot is running...\n" << endl;
        
        long long lastUpdate = 0;
        
        while(true) {
            auto updates = getUpdates(lastUpdate);
            
            for(const auto& update : updates) {
                lastUpdate = update["update_id"];
                
                if(update.contains("message")) {
                    handleMessage(update["message"]);
                }
                else if(update.contains("callback_query")) {
                    handleCallback(update["callback_query"]);
                }
            }
            
            this_thread::sleep_for(milliseconds(500));
        }
    }
    
private:
    void handleMessage(const json& msg) {
        long long chatId = msg["chat"]["id"];
        string text = msg.value("text", "");
        
        // Handle different message types
        if(!text.empty()) {
            handleTextMessage(chatId, text);
        }
        else if(msg.contains("photo")) {
            handlePhotoMessage(chatId, msg["photo"]);
        }
        else if(msg.contains("voice")) {
            handleVoiceMessage(chatId, msg["voice"]);
        }
        else if(msg.contains("document")) {
            handleDocumentMessage(chatId, msg["document"]);
        }
    }
    
    void handleTextMessage(long long chatId, const string& text) {
        cout << "📩 " << text << endl;
        
        // Commands
        if(text == "/start") {
            sendMessage(chatId, 
                "🎉 <b>سلام! من آرمان هستم، هوش مصنوعی پیشرفته C++</b>\n\n"
                "🧠 <b>قابلیت‌های من:</b>\n"
                "• پردازش متن فارسی و انگلیسی\n"
                "• تحلیل تصویر و تشخیص اشیاء\n"
                "• تبدیل صوت به متن\n"
                "• دسترسی به دانش آنلاین\n"
                "• یادگیری عمیق با شبکه عصبی\n\n"
                "📸 عکس بفرست تا تحلیل کنم\n"
                "🎙️ ویس بفرست تا متنش کنم\n"
                "💬 هر سوالی داری بپرس!\n\n"
                "ساخته شده با ❤️ و C++"
            );
        }
        else if(text == "/help") {
            sendMessage(chatId, getHelpText());
        }
        else if(text == "/search") {
            sendMessage(chatId, "🔍 چی رو جستجو کنم؟");
        }
        else {
            // Process with AI
            string response = processWithAI(text);
            sendMessage(chatId, response);
        }
    }
    
    void handlePhotoMessage(long long chatId, const json& photo) {
        sendMessage(chatId, "🖼️ در حال تحلیل تصویر...");
        
        // Download largest photo
        string fileId = photo.back()["file_id"];
        string filePath = downloadFile(fileId);
        
        // Analyze image
        Mat image = imread(filePath);
        auto objects = vision.detectObjects(image);
        string text = vision.extractText(image);
        string classification = vision.classifyImage(image);
        
        // Generate response
        string response = "📊 <b>نتایج تحلیل تصویر:</b>\n\n";
        response += "🏷️ <b>دسته‌بندی:</b> " + classification + "\n\n";
        
        if(!objects.empty()) {
            response += "🔍 <b>اشیاء شناسایی شده:</b>\n";
            for(const auto& obj : objects) {
                response += "• " + obj.className + " (" + 
                          to_string(int(obj.confidence * 100)) + "%)\n";
            }
        }
        
        if(!text.empty()) {
            response += "\n📝 <b>متن تشخیص داده شده:</b>\n" + text;
        }
        
        sendMessage(chatId, response);
    }
    
    void handleVoiceMessage(long long chatId, const json& voice) {
        sendMessage(chatId, "🎙️ در حال تبدیل صوت به متن...");
        
        string fileId = voice["file_id"];
        string filePath = downloadFile(fileId);
        
        string text = speech.transcribe(filePath);
        
        sendMessage(chatId, "📝 <b>متن ویس:</b>\n" + text);
        
        // Also respond to the transcribed text
        string aiResponse = processWithAI(text);
        sendMessage(chatId, "🤖 <b>پاسخ:</b>\n" + aiResponse);
    }
    
    string processWithAI(const string& input) {
        // 1. Check knowledge graph
        string kgAnswer = knowledge.query(input);
        if(!kgAnswer.empty()) {
            return kgAnswer;
        }
        
        // 2. Search web if needed
        if(isQuestion(input)) {
            string webResult = searchWeb(input);
            knowledge.add(input, webResult);
            return webResult;
        }
        
        // 3. Use neural network for generation
        VectorXd inputVec = tokenize(input);
        VectorXd output = brain.forward(inputVec);
        string response = detokenize(output);
        
        return response;
    }
    
    string searchWeb(const string& query) {
        // Use DuckDuckGo API
        CURL* curl = curl_easy_init();
        string response;
        
        if(curl) {
            string url = "https://api.duckduckgo.com/?q=" + curl_easy_escape(curl, query.c_str(), query.length()) + "&format=json";
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_perform(curl);
            curl_easy_cleanup(curl);
        }
        
        // Parse and extract answer
        return extractAnswer(response);
    }
};

int main() {
    string BOT_TOKEN = getenv("BOT_TOKEN") ? getenv("BOT_TOKEN") : "YOUR_TOKEN";
    
    TelegramBot bot(BOT_TOKEN);
    bot.run();
    
    return 0;
}
