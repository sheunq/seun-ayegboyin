import { AboutSection } from "@/components/home/AboutSection";
import { BlogSection } from "@/components/home/BlogSection";
import { HeroSection } from "@/components/home/HeroSection";
import { ProjectsSection } from "@/components/home/ProjectsSection";
import { TechnicalSkillsSection } from "@/components/home/TechnicalSkillsSection";
import { getBlogPosts, getProjectCategories } from "@/lib/data";
import { JourneySection } from "@/components/home/JourneySection";

export default function Home() {
  const projectCategories = getProjectCategories();
  const blogPosts = getBlogPosts();

  return (
    <div className="flex flex-col gap-16 md:gap-24">
      <HeroSection />
      <AboutSection />
      <ProjectsSection categories={projectCategories} />
      <TechnicalSkillsSection />
      <JourneySection />
      <BlogSection posts={blogPosts.slice(0, 3)} />
    </div>
  );
}
